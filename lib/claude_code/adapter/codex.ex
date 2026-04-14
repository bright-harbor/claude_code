defmodule ClaudeCode.Adapter.Codex do
  @moduledoc """
  Experimental adapter that drives OpenAI's `codex` CLI as a backend for the
  ClaudeCode SDK.

  Instead of spawning `claude` and speaking Anthropic's stream-json protocol,
  this adapter spawns `codex app-server --transport stdio` and speaks codex's
  JSON-RPC 2.0 protocol. Incoming codex notifications are translated on the
  fly into `ClaudeCode.Message.*` and `ClaudeCode.Content.*` structs and
  delivered to `Session` via `Adapter.notify_message/3` — the same path
  `Adapter.Test` uses — so the rest of the SDK (streams, typed messages,
  `ResultMessage` auto-completion) is unchanged.

  This is a minimum-viable implementation covering the happy path:

  - Session start (`thread/start`) → synthesized `SystemMessage` with
    `subtype: :init`
  - User turns (`turn/start`) with optional `system_prompt` prepended
  - Streaming text via `PartialAssistantMessage` (`text_delta`)
  - Streaming reasoning via `PartialAssistantMessage` (`thinking_delta`)
  - Completed agent messages as `AssistantMessage` with a `TextBlock`
  - Command execution / MCP tool calls as synthetic
    `ToolUseBlock`+`ToolResultBlock` pairs
  - Turn completion → `ResultMessage`, failure → error `ResultMessage`
  - `interrupt/1` via `turn/interrupt`

  Not implemented yet:

  - `can_use_tool` / approval round-tripping (codex handles tools itself in
    `--full-auto`; add a `ServerRequest` bridge to change this)
  - Hook invocation
  - MCP config translation from `--mcp-config`
  - Cost estimation (`total_cost_usd` is stubbed to `0.0`)

  ## Usage

      {:ok, session} =
        ClaudeCode.start_link(
          adapter:
            {ClaudeCode.Adapter.Codex,
             codex_path: "codex",
             model: "gpt-5-codex",
             cwd: File.cwd!()}
        )

      ClaudeCode.query(session, "summarize this repo")

  ## Config keys

  - `:codex_path` — path to the `codex` binary (default: `"codex"`, resolved
    from `$PATH`)
  - `:model` — passed to `thread/start` params
  - `:cwd` — working directory for the codex subprocess
  - `:system_prompt` / `:append_system_prompt` — prepended to the first turn
    as a `<system>` block
  - `:codex_env` — additional environment variables (map) for codex
  """

  @behaviour ClaudeCode.Adapter

  use GenServer

  alias ClaudeCode.Adapter
  alias ClaudeCode.Content.TextBlock
  alias ClaudeCode.Content.ThinkingBlock
  alias ClaudeCode.Content.ToolResultBlock
  alias ClaudeCode.Content.ToolUseBlock
  alias ClaudeCode.Message.AssistantMessage
  alias ClaudeCode.Message.PartialAssistantMessage
  alias ClaudeCode.Message.ResultMessage
  alias ClaudeCode.Message.SystemMessage
  alias ClaudeCode.Message.UserMessage

  require Logger

  defstruct [
    :session,
    :session_options,
    :port,
    :thread_id,
    :current_request,
    :turn_started_at,
    :pending_system_prompt,
    :pending_init_message,
    buffer: "",
    status: :provisioning,
    rpc_counter: 0,
    pending_rpc: %{},
    block_indexes: %{},
    next_index: 0,
    num_turns: 0
  ]

  # ============================================================================
  # Client API (Adapter Behaviour)
  # ============================================================================

  @impl ClaudeCode.Adapter
  def start_link(session, opts) do
    GenServer.start_link(__MODULE__, {session, opts})
  end

  @impl ClaudeCode.Adapter
  def send_query(adapter, request_id, prompt, opts) do
    GenServer.call(adapter, {:query, request_id, prompt, opts}, :infinity)
  end

  @impl ClaudeCode.Adapter
  def health(adapter), do: GenServer.call(adapter, :health)

  @impl ClaudeCode.Adapter
  def stop(adapter), do: GenServer.stop(adapter, :normal)

  @impl ClaudeCode.Adapter
  def interrupt(adapter), do: GenServer.call(adapter, :interrupt)

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl GenServer
  def init({session, opts}) do
    state = %__MODULE__{
      session: session,
      session_options: opts,
      pending_system_prompt: build_system_prompt(opts)
    }

    Process.link(session)
    Adapter.notify_status(session, :provisioning)

    {:ok, state, {:continue, :connect}}
  end

  @impl GenServer
  def handle_continue(:connect, state) do
    case open_codex_port(state) do
      {:ok, port} ->
        state = %{state | port: port}
        # Kick off thread/start immediately so Session transitions to :ready
        # once we receive the response and synthesize the init message.
        {_id, state} =
          send_rpc(state, "thread/start", thread_start_params(state), :thread_start)

        {:noreply, state}

      {:error, reason} ->
        Adapter.notify_status(state.session, {:error, reason})
        {:stop, {:shutdown, reason}, state}
    end
  end

  @impl GenServer
  def handle_call({:query, request_id, prompt, opts}, _from, state) do
    session_id = state.thread_id || Keyword.get(opts, :session_id, "codex")

    full_prompt = prepend_system_prompt(prompt, state.pending_system_prompt)

    state = %{
      state
      | current_request: request_id,
        turn_started_at: System.monotonic_time(:millisecond),
        num_turns: state.num_turns + 1,
        block_indexes: %{},
        next_index: 0,
        pending_system_prompt: nil
    }

    # Claude emits the system/init message at the start of the first turn's
    # stream, not at handshake time. We cached it on thread/start; flush it
    # now so consumers see it on the same request stream.
    state = flush_pending_init(state)

    params = %{
      "threadId" => session_id,
      "input" => [%{"type" => "text", "text" => full_prompt, "textElements" => []}]
    }

    {_id, state} = send_rpc(state, "turn/start", params, :turn_start)
    {:reply, :ok, state}
  end

  def handle_call(:health, _from, state) do
    health =
      case state.status do
        :ready -> :healthy
        :provisioning -> {:unhealthy, :provisioning}
        other -> {:unhealthy, other}
      end

    {:reply, health, state}
  end

  def handle_call(:interrupt, _from, state) do
    case state.thread_id do
      nil ->
        {:reply, :ok, state}

      tid ->
        {_, state} = send_rpc(state, "turn/interrupt", %{"threadId" => tid}, :interrupt)
        {:reply, :ok, state}
    end
  end

  @impl GenServer
  def handle_info({port, {:data, chunk}}, %{port: port} = state) do
    {lines, rest} = extract_lines(state.buffer <> chunk)
    state = %{state | buffer: rest}
    state = Enum.reduce(lines, state, &process_line/2)
    {:noreply, state}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("codex app-server exited with status #{status}")
    Adapter.notify_status(state.session, {:error, {:codex_exit, status}})
    {:stop, {:shutdown, {:codex_exit, status}}, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl GenServer
  def terminate(_reason, %{port: port} = state) when is_port(port) do
    if state.thread_id do
      # Best-effort interrupt; ignore failures on shutdown.
      line = encode_rpc(0, "turn/interrupt", %{"threadId" => state.thread_id})
      send_port(port, line)
    end

    Port.close(port)
    :ok
  rescue
    _ -> :ok
  end

  def terminate(_reason, _state), do: :ok

  # ============================================================================
  # Line processing
  # ============================================================================

  @doc false
  def extract_lines(buffer) do
    case String.split(buffer, "\n") do
      [incomplete] -> {[], incomplete}
      lines -> {List.delete_at(lines, -1), List.last(lines)}
    end
  end

  defp process_line("", state), do: state

  defp process_line(line, state) do
    case Jason.decode(line) do
      {:ok, json} -> dispatch(json, state)
      {:error, _} -> state
    end
  end

  # JSON-RPC response: either a thread/start ack or another call ack.
  defp dispatch(%{"id" => id, "result" => result}, state) do
    case Map.pop(state.pending_rpc, id) do
      {:thread_start, rest} ->
        thread_id = get_in(result, ["thread", "id"]) || result["threadId"] || "codex-thread"

        # Cache the init message until the first user turn begins, because
        # Session drops adapter_message for a request_id it doesn't know
        # about. Matches the Claude CLI, which emits system/init inside
        # the first turn's stream, not at handshake time.
        init_msg = build_init_message(thread_id, state)
        Adapter.notify_status(state.session, :ready)

        %{
          state
          | thread_id: thread_id,
            pending_rpc: rest,
            status: :ready,
            pending_init_message: init_msg
        }

      {_tag_or_nil, rest} ->
        # Ack for turn/start, turn/interrupt, or unknown id — nothing to do
        # beyond clearing the pending entry.
        %{state | pending_rpc: rest}
    end
  end

  defp dispatch(%{"id" => id, "error" => error}, state) do
    Logger.warning("codex RPC error: #{inspect(error)}")
    %{state | pending_rpc: Map.delete(state.pending_rpc, id)}
  end

  # JSON-RPC notification: the interesting stream.
  defp dispatch(%{"method" => method, "params" => params}, state),
    do: translate(method, params, state)

  defp dispatch(_, state), do: state

  # ============================================================================
  # Notification → Claude message translation
  # ============================================================================

  # Agent text deltas — character-level streaming.
  defp translate("agentMessageDelta", %{"itemId" => item_id, "delta" => delta}, state) do
    {idx, state} = index_for(state, item_id)
    emit_stream_event(state, content_block_delta(idx, %{type: :text_delta, text: delta}))
  end

  defp translate("reasoningTextDelta", %{"itemId" => item_id, "delta" => delta}, state) do
    {idx, state} = index_for(state, {:thinking, item_id})
    emit_stream_event(state, content_block_delta(idx, %{type: :thinking_delta, thinking: delta}))
  end

  defp translate("reasoningSummaryTextDelta", %{"itemId" => item_id, "delta" => delta}, state) do
    {idx, state} = index_for(state, {:thinking, item_id})
    emit_stream_event(state, content_block_delta(idx, %{type: :thinking_delta, thinking: delta}))
  end

  # Item lifecycle — agent_message: announce a text block, on completion emit
  # a proper AssistantMessage with the finished TextBlock.
  defp translate("itemStarted", %{"item" => %{"type" => "agent_message", "id" => item_id}}, state) do
    {idx, state} = index_for(state, item_id)
    emit_stream_event(state, content_block_start(idx, %{type: :text, text: ""}))
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "agent_message", "id" => item_id, "text" => text}},
         state
       ) do
    {idx, state} = index_for(state, item_id)
    state = emit_stream_event(state, content_block_stop(idx))

    msg = %AssistantMessage{
      type: :assistant,
      session_id: state.thread_id,
      message: %{
        id: item_id,
        type: :message,
        role: :assistant,
        content: [%TextBlock{type: :text, text: text}],
        model: nil,
        stop_reason: :end_turn,
        stop_sequence: nil,
        usage: empty_usage(),
        context_management: nil
      }
    }

    notify(state, msg)
  end

  # Reasoning items — emit a final ThinkingBlock when complete.
  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "reasoning", "id" => item_id, "text" => text}},
         state
       ) do
    {idx, state} = index_for(state, {:thinking, item_id})
    state = emit_stream_event(state, content_block_stop(idx))

    msg = %AssistantMessage{
      type: :assistant,
      session_id: state.thread_id,
      message: %{
        id: item_id,
        type: :message,
        role: :assistant,
        content: [%ThinkingBlock{type: :thinking, thinking: text, signature: ""}],
        model: nil,
        stop_reason: nil,
        stop_sequence: nil,
        usage: empty_usage(),
        context_management: nil
      }
    }

    notify(state, msg)
  end

  # Command execution — fabricate a tool_use/tool_result pair so
  # ClaudeCode.Stream.tool_uses/1 keeps working as a read-only view of what
  # codex actually ran.
  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "command_execution"} = item},
         state
       ) do
    tool_use_id = "codex_cmd_#{item["id"]}"

    use_msg = %AssistantMessage{
      type: :assistant,
      session_id: state.thread_id,
      message: %{
        id: item["id"],
        type: :message,
        role: :assistant,
        content: [
          %ToolUseBlock{
            type: :tool_use,
            id: tool_use_id,
            name: "Bash",
            input: %{"command" => item["command"]}
          }
        ],
        model: nil,
        stop_reason: :tool_use,
        stop_sequence: nil,
        usage: empty_usage(),
        context_management: nil
      }
    }

    result_msg = %UserMessage{
      type: :user,
      session_id: state.thread_id,
      message: %{
        role: :user,
        content: [
          %ToolResultBlock{
            type: :tool_result,
            tool_use_id: tool_use_id,
            content: item["aggregated_output"] || "",
            is_error: (item["exit_code"] || 0) != 0
          }
        ]
      }
    }

    state |> notify(use_msg) |> notify(result_msg)
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "mcp_tool_call"} = item},
         state
       ) do
    tool_use_id = "codex_mcp_#{item["id"]}"
    tool_name = "#{item["server"] || "mcp"}__#{item["tool"] || "unknown"}"

    use_msg = %AssistantMessage{
      type: :assistant,
      session_id: state.thread_id,
      message: %{
        id: item["id"],
        type: :message,
        role: :assistant,
        content: [
          %ToolUseBlock{
            type: :tool_use,
            id: tool_use_id,
            name: tool_name,
            input: item["arguments"] || %{}
          }
        ],
        model: nil,
        stop_reason: :tool_use,
        stop_sequence: nil,
        usage: empty_usage(),
        context_management: nil
      }
    }

    result_msg = %UserMessage{
      type: :user,
      session_id: state.thread_id,
      message: %{
        role: :user,
        content: [
          %ToolResultBlock{
            type: :tool_result,
            tool_use_id: tool_use_id,
            content: format_mcp_result(item),
            is_error: item["error"] != nil
          }
        ]
      }
    }

    state |> notify(use_msg) |> notify(result_msg)
  end

  # Unhandled item types: silently ignore (file_change, web_search, todo_list
  # etc. would land here). The stream still works, we just don't surface them.
  defp translate("itemStarted", _params, state), do: state
  defp translate("itemUpdated", _params, state), do: state
  defp translate("itemCompleted", _params, state), do: state

  defp translate("turnStarted", _params, state), do: state

  defp translate("turnCompleted", %{"usage" => usage}, state) do
    duration = System.monotonic_time(:millisecond) - (state.turn_started_at || 0)

    msg = %ResultMessage{
      type: :result,
      subtype: :success,
      is_error: false,
      duration_ms: duration * 1.0,
      duration_api_ms: duration * 1.0,
      num_turns: state.num_turns,
      session_id: state.thread_id,
      total_cost_usd: 0.0,
      usage: codex_usage_to_claude(usage),
      result: nil,
      stop_reason: :end_turn,
      model_usage: %{},
      permission_denials: [],
      errors: nil
    }

    state = notify(state, msg)
    %{state | current_request: nil}
  end

  defp translate("turnFailed", %{"error" => error}, state) do
    duration = System.monotonic_time(:millisecond) - (state.turn_started_at || 0)

    msg = %ResultMessage{
      type: :result,
      subtype: :error_during_execution,
      is_error: true,
      duration_ms: duration * 1.0,
      duration_api_ms: duration * 1.0,
      num_turns: state.num_turns,
      session_id: state.thread_id,
      total_cost_usd: 0.0,
      usage: empty_result_usage(),
      result: error["message"],
      stop_reason: nil,
      model_usage: %{},
      permission_denials: [],
      errors: [error["message"] || "codex turn failed"]
    }

    state = notify(state, msg)
    %{state | current_request: nil}
  end

  defp translate(_method, _params, state), do: state

  # ============================================================================
  # Small helpers
  # ============================================================================

  defp notify(%{current_request: nil} = state, _msg), do: state

  defp notify(state, msg) do
    Adapter.notify_message(state.session, state.current_request, msg)
    state
  end

  defp flush_pending_init(%{pending_init_message: nil} = state), do: state

  defp flush_pending_init(%{pending_init_message: init} = state) do
    Adapter.notify_message(state.session, state.current_request, init)
    %{state | pending_init_message: nil}
  end

  defp emit_stream_event(%{current_request: nil} = state, _event), do: state

  defp emit_stream_event(state, event) do
    partial = %PartialAssistantMessage{
      type: :stream_event,
      event: event,
      session_id: state.thread_id
    }

    Adapter.notify_message(state.session, state.current_request, partial)
    state
  end

  defp content_block_start(index, block),
    do: %{type: :content_block_start, index: index, content_block: block}

  defp content_block_delta(index, delta),
    do: %{type: :content_block_delta, index: index, delta: delta}

  defp content_block_stop(index),
    do: %{type: :content_block_stop, index: index}

  defp index_for(state, key) do
    case Map.fetch(state.block_indexes, key) do
      {:ok, idx} ->
        {idx, state}

      :error ->
        idx = state.next_index

        {idx,
         %{
           state
           | block_indexes: Map.put(state.block_indexes, key, idx),
             next_index: idx + 1
         }}
    end
  end

  defp build_init_message(thread_id, state) do
    model = Keyword.get(state.session_options, :model) || "codex"

    %SystemMessage{
      type: :system,
      subtype: :init,
      session_id: thread_id,
      cwd: Keyword.get(state.session_options, :cwd),
      tools: ["Bash", "Read", "Write", "Edit", "WebSearch"],
      mcp_servers: [],
      model: model,
      permission_mode: :default,
      api_key_source: "codex",
      claude_code_version: nil
    }
  end

  defp thread_start_params(state) do
    opts = state.session_options

    base = %{}
    base = maybe_put(base, "model", Keyword.get(opts, :model))
    base = maybe_put(base, "cwd", Keyword.get(opts, :cwd))
    base
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  defp build_system_prompt(opts) do
    case {Keyword.get(opts, :system_prompt), Keyword.get(opts, :append_system_prompt)} do
      {nil, nil} -> nil
      {sp, nil} -> sp
      {nil, ap} -> ap
      {sp, ap} -> sp <> "\n\n" <> ap
    end
  end

  defp prepend_system_prompt(prompt, nil), do: prompt

  defp prepend_system_prompt(prompt, system),
    do: "<system>\n" <> system <> "\n</system>\n\n" <> prompt

  defp codex_usage_to_claude(%{} = usage) do
    %{
      input_tokens: usage["input_tokens"] || usage["inputTokens"] || 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: usage["cached_input_tokens"] || usage["cachedInputTokens"] || 0,
      output_tokens: usage["output_tokens"] || usage["outputTokens"] || 0,
      server_tool_use: %{web_search_requests: 0, web_fetch_requests: 0},
      service_tier: nil,
      cache_creation: nil,
      inference_geo: nil,
      iterations: [],
      speed: nil
    }
  end

  defp codex_usage_to_claude(_), do: empty_result_usage()

  defp empty_result_usage do
    %{
      input_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      output_tokens: 0,
      server_tool_use: %{web_search_requests: 0, web_fetch_requests: 0},
      service_tier: nil,
      cache_creation: nil,
      inference_geo: nil,
      iterations: [],
      speed: nil
    }
  end

  defp empty_usage do
    %{
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: nil,
      cache_read_input_tokens: nil,
      cache_creation: nil,
      service_tier: nil,
      inference_geo: nil
    }
  end

  defp format_mcp_result(%{"error" => %{"message" => msg}}) when is_binary(msg), do: msg

  defp format_mcp_result(%{"result" => %{"content" => content}}) when is_list(content) do
    content
    |> Enum.map_join("\n", fn
      %{"text" => text} when is_binary(text) -> text
      other -> inspect(other)
    end)
  end

  defp format_mcp_result(_), do: ""

  # ============================================================================
  # JSON-RPC I/O
  # ============================================================================

  defp send_rpc(state, method, params, tag) do
    id = state.rpc_counter + 1
    line = encode_rpc(id, method, params)
    send_port(state.port, line)

    state = %{
      state
      | rpc_counter: id,
        pending_rpc: Map.put(state.pending_rpc, id, tag)
    }

    {id, state}
  end

  defp encode_rpc(id, method, params) do
    Jason.encode!(%{"jsonrpc" => "2.0", "id" => id, "method" => method, "params" => params}) <>
      "\n"
  end

  defp send_port(port, line) when is_port(port) do
    try do
      Port.command(port, line)
    rescue
      _ -> :error
    end
  end

  # ============================================================================
  # Port spawning
  # ============================================================================

  defp open_codex_port(state) do
    codex_path = Keyword.get(state.session_options, :codex_path, "codex")

    case resolve_executable(codex_path) do
      nil ->
        {:error, {:codex_not_found, codex_path}}

      exe ->
        args = ["app-server", "--transport", "stdio"]

        try do
          port =
            Port.open({:spawn_executable, exe}, [
              {:args, args},
              :binary,
              :exit_status,
              :stderr_to_stdout
            ])

          {:ok, port}
        rescue
          e -> {:error, {:port_open_failed, Exception.message(e)}}
        end
    end
  end

  defp resolve_executable(path) when is_binary(path) do
    if Path.type(path) == :absolute and File.exists?(path) do
      path
    else
      case :os.find_executable(String.to_charlist(path)) do
        false -> nil
        exe -> List.to_string(exe)
      end
    end
  end

  defp resolve_executable(_), do: nil

  # ============================================================================
  # Test seams
  # ============================================================================

  @doc false
  # Public entry point used by tests to feed a decoded JSON-RPC payload
  # through the same dispatcher the live Port path uses.
  def __dispatch__(json, state), do: dispatch(json, state)

  @doc false
  def __new_state__(session, opts \\ []) do
    %__MODULE__{
      session: session,
      session_options: opts,
      pending_system_prompt: build_system_prompt(opts),
      thread_id: Keyword.get(opts, :thread_id),
      current_request: Keyword.get(opts, :current_request),
      turn_started_at: System.monotonic_time(:millisecond),
      num_turns: Keyword.get(opts, :num_turns, 1),
      status: :ready
    }
  end
end
