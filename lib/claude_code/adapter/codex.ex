defmodule ClaudeCode.Adapter.Codex do
  @moduledoc """
  Experimental adapter that drives OpenAI's `codex` CLI as a backend for the
  ClaudeCode SDK.

  Spawns `codex app-server --transport stdio` and translates its JSON-RPC
  notifications into `ClaudeCode.Message.*` structs in-process — the same
  delivery path `Adapter.Test` uses — so streams, typed messages, and
  `ResultMessage` auto-completion keep working unchanged.

  Covers streaming text + reasoning, command / MCP tool visibility, turn
  boundaries, and `interrupt/1`. `can_use_tool`, hooks, MCP config
  translation, and cost estimation are intentionally out of scope.

  ## Usage

      {:ok, session} =
        ClaudeCode.start_link(
          adapter:
            {ClaudeCode.Adapter.Codex,
             codex_path: "codex", model: "gpt-5-codex", cwd: File.cwd!()}
        )

  Config keys (inside the adapter tuple): `:codex_path`, `:model`, `:cwd`,
  `:system_prompt`, `:append_system_prompt`.
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

  # --- Adapter behaviour ---

  @impl ClaudeCode.Adapter
  def start_link(session, opts), do: GenServer.start_link(__MODULE__, {session, opts})

  @impl ClaudeCode.Adapter
  def send_query(adapter, request_id, prompt, opts),
    do: GenServer.call(adapter, {:query, request_id, prompt, opts}, :infinity)

  @impl ClaudeCode.Adapter
  def health(adapter), do: GenServer.call(adapter, :health)

  @impl ClaudeCode.Adapter
  def stop(adapter), do: GenServer.stop(adapter, :normal)

  @impl ClaudeCode.Adapter
  def interrupt(adapter), do: GenServer.call(adapter, :interrupt)

  # --- GenServer callbacks ---

  @impl GenServer
  def init({session, opts}) do
    Process.link(session)
    Adapter.notify_status(session, :provisioning)

    {:ok,
     %__MODULE__{
       session: session,
       session_options: opts,
       pending_system_prompt: build_system_prompt(opts)
     }, {:continue, :connect}}
  end

  @impl GenServer
  def handle_continue(:connect, state) do
    case open_codex_port(state) do
      {:ok, port} ->
        state = %{state | port: port}
        {_id, state} = send_rpc(state, "thread/start", thread_start_params(state), :thread_start)
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

    state =
      %{
        state
        | current_request: request_id,
          turn_started_at: System.monotonic_time(:millisecond),
          num_turns: state.num_turns + 1,
          block_indexes: %{},
          next_index: 0,
          pending_system_prompt: nil
      }
      |> flush_pending_init()

    params = %{
      "threadId" => session_id,
      "input" => [%{"type" => "text", "text" => full_prompt, "textElements" => []}]
    }

    {_id, state} = send_rpc(state, "turn/start", params, :turn_start)
    {:reply, :ok, state}
  end

  def handle_call(:health, _from, %{status: :ready} = state), do: {:reply, :healthy, state}

  def handle_call(:health, _from, %{status: status} = state),
    do: {:reply, {:unhealthy, status}, state}

  def handle_call(:interrupt, _from, %{thread_id: nil} = state), do: {:reply, :ok, state}

  def handle_call(:interrupt, _from, %{thread_id: tid} = state) do
    {_, state} = send_rpc(state, "turn/interrupt", %{"threadId" => tid}, :interrupt)
    {:reply, :ok, state}
  end

  @impl GenServer
  def handle_info({port, {:data, chunk}}, %{port: port} = state) do
    {lines, rest} = extract_lines(state.buffer <> chunk)
    {:noreply, Enum.reduce(lines, %{state | buffer: rest}, &process_line/2)}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("codex app-server exited with status #{status}")
    Adapter.notify_status(state.session, {:error, {:codex_exit, status}})
    {:stop, {:shutdown, {:codex_exit, status}}, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  @impl GenServer
  def terminate(_reason, %{port: port, thread_id: tid} = _state) when is_port(port) do
    if tid, do: Port.command(port, encode_rpc(0, "turn/interrupt", %{"threadId" => tid}))
    Port.close(port)
    :ok
  rescue
    _ -> :ok
  end

  def terminate(_, _), do: :ok

  # --- Line processing ---

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

  # Response to one of our own JSON-RPC requests.
  defp dispatch(%{"id" => id, "result" => result}, state) do
    case Map.pop(state.pending_rpc, id) do
      {:thread_start, rest} ->
        thread_id = get_in(result, ["thread", "id"]) || result["threadId"] || "codex-thread"
        Adapter.notify_status(state.session, :ready)

        %{
          state
          | thread_id: thread_id,
            pending_rpc: rest,
            status: :ready,
            pending_init_message: build_init_message(thread_id, state)
        }

      {_tag_or_nil, rest} ->
        %{state | pending_rpc: rest}
    end
  end

  defp dispatch(%{"id" => id, "error" => error}, state) do
    Logger.warning("codex RPC error: #{inspect(error)}")
    %{state | pending_rpc: Map.delete(state.pending_rpc, id)}
  end

  defp dispatch(%{"method" => method, "params" => params}, state),
    do: translate(method, params, state)

  defp dispatch(_, state), do: state

  # --- Codex notification -> Claude message translation ---

  defp translate("agentMessageDelta", %{"itemId" => id, "delta" => d}, state) do
    {idx, state} = index_for(state, id)
    emit_stream_event(state, content_block_delta(idx, %{type: :text_delta, text: d}))
  end

  defp translate(method, %{"itemId" => id, "delta" => d}, state)
       when method in ~w(reasoningTextDelta reasoningSummaryTextDelta) do
    {idx, state} = index_for(state, {:thinking, id})
    emit_stream_event(state, content_block_delta(idx, %{type: :thinking_delta, thinking: d}))
  end

  defp translate("itemStarted", %{"item" => %{"type" => "agent_message", "id" => id}}, state) do
    {idx, state} = index_for(state, id)
    emit_stream_event(state, content_block_start(idx, %{type: :text, text: ""}))
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "agent_message", "id" => id, "text" => text}},
         state
       ) do
    {idx, state} = index_for(state, id)
    msg = assistant_message(state, id, [text_block(text)], :end_turn)
    state |> emit_stream_event(content_block_stop(idx)) |> notify(msg)
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "reasoning", "id" => id, "text" => text}},
         state
       ) do
    {idx, state} = index_for(state, {:thinking, id})
    msg = assistant_message(state, id, [thinking_block(text)], nil)
    state |> emit_stream_event(content_block_stop(idx)) |> notify(msg)
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "command_execution", "id" => id, "command" => command} = item},
         state
       ) do
    emit_tool_pair(state, %{
      item_id: id,
      tool_use_id: "codex_cmd_#{id}",
      name: "Bash",
      input: %{"command" => command},
      content: item["aggregated_output"] || "",
      is_error: (item["exit_code"] || 0) != 0
    })
  end

  defp translate(
         "itemCompleted",
         %{"item" => %{"type" => "mcp_tool_call", "id" => id} = item},
         state
       ) do
    emit_tool_pair(state, %{
      item_id: id,
      tool_use_id: "codex_mcp_#{id}",
      name: "#{item["server"] || "mcp"}__#{item["tool"] || "unknown"}",
      input: item["arguments"] || %{},
      content: format_mcp_result(item),
      is_error: item["error"] != nil
    })
  end

  defp translate("turnCompleted", %{"usage" => usage}, state) do
    state
    |> notify(
      result_message(state,
        subtype: :success,
        is_error: false,
        usage: codex_usage_to_claude(usage),
        stop_reason: :end_turn
      )
    )
    |> clear_request()
  end

  defp translate("turnFailed", %{"error" => %{"message" => message}}, state),
    do: fail_turn(state, message)

  defp translate("turnFailed", _params, state), do: fail_turn(state, "codex turn failed")

  # Everything else (itemStarted/itemUpdated for other item types, turnStarted,
  # unknown future methods) is silently ignored.
  defp translate(_method, _params, state), do: state

  # --- Message builders ---

  defp assistant_message(state, id, content, stop_reason) do
    %AssistantMessage{
      type: :assistant,
      session_id: state.thread_id,
      message: %{
        id: id,
        type: :message,
        role: :assistant,
        content: content,
        model: nil,
        stop_reason: stop_reason,
        stop_sequence: nil,
        usage: base_usage(),
        context_management: nil
      }
    }
  end

  defp user_message(state, content) do
    %UserMessage{
      type: :user,
      session_id: state.thread_id,
      message: %{role: :user, content: content}
    }
  end

  defp result_message(state, attrs) do
    duration = System.monotonic_time(:millisecond) - (state.turn_started_at || 0)

    base = [
      type: :result,
      duration_ms: duration * 1.0,
      duration_api_ms: duration * 1.0,
      num_turns: state.num_turns,
      session_id: state.thread_id,
      total_cost_usd: 0.0,
      usage: base_usage(),
      result: nil,
      stop_reason: nil,
      model_usage: %{},
      permission_denials: [],
      errors: nil
    ]

    struct!(ResultMessage, Keyword.merge(base, attrs))
  end

  defp text_block(text), do: %TextBlock{type: :text, text: text}

  defp thinking_block(text),
    do: %ThinkingBlock{type: :thinking, thinking: text, signature: ""}

  defp tool_use_block(id, name, input),
    do: %ToolUseBlock{type: :tool_use, id: id, name: name, input: input}

  defp tool_result_block(tool_use_id, content, is_error),
    do: %ToolResultBlock{
      type: :tool_result,
      tool_use_id: tool_use_id,
      content: content,
      is_error: is_error
    }

  defp fail_turn(state, message) do
    state
    |> notify(
      result_message(state,
        subtype: :error_during_execution,
        is_error: true,
        result: message,
        errors: [message]
      )
    )
    |> clear_request()
  end

  defp emit_tool_pair(state, %{
         item_id: item_id,
         tool_use_id: tool_use_id,
         name: name,
         input: input,
         content: content,
         is_error: is_error
       }) do
    state
    |> notify(
      assistant_message(state, item_id, [tool_use_block(tool_use_id, name, input)], :tool_use)
    )
    |> notify(user_message(state, [tool_result_block(tool_use_id, content, is_error)]))
  end

  defp clear_request(state), do: %{state | current_request: nil}

  # --- Small helpers ---

  defp notify(%{current_request: nil} = state, _msg), do: state

  defp notify(state, msg) do
    Adapter.notify_message(state.session, state.current_request, msg)
    state
  end

  defp flush_pending_init(%{pending_init_message: nil} = state), do: state

  defp flush_pending_init(%{pending_init_message: init} = state) do
    state |> notify(init) |> Map.put(:pending_init_message, nil)
  end

  defp emit_stream_event(state, event) do
    notify(state, %PartialAssistantMessage{
      type: :stream_event,
      event: event,
      session_id: state.thread_id
    })
  end

  defp content_block_start(index, block),
    do: %{type: :content_block_start, index: index, content_block: block}

  defp content_block_delta(index, delta),
    do: %{type: :content_block_delta, index: index, delta: delta}

  defp content_block_stop(index), do: %{type: :content_block_stop, index: index}

  defp index_for(%{block_indexes: map} = state, key) do
    case map do
      %{^key => idx} ->
        {idx, state}

      _ ->
        idx = state.next_index
        {idx, %{state | block_indexes: Map.put(map, key, idx), next_index: idx + 1}}
    end
  end

  defp build_init_message(thread_id, state) do
    %SystemMessage{
      type: :system,
      subtype: :init,
      session_id: thread_id,
      cwd: state.session_options[:cwd],
      tools: ~w(Bash Read Write Edit WebSearch),
      mcp_servers: [],
      model: state.session_options[:model] || "codex",
      permission_mode: :default,
      api_key_source: "codex",
      claude_code_version: nil
    }
  end

  defp thread_start_params(state) do
    state.session_options
    |> Keyword.take([:model, :cwd])
    |> Enum.reject(fn {_, v} -> is_nil(v) end)
    |> Map.new(fn {k, v} -> {to_string(k), v} end)
  end

  defp build_system_prompt(opts) do
    case Enum.reject([opts[:system_prompt], opts[:append_system_prompt]], &is_nil/1) do
      [] -> nil
      parts -> Enum.join(parts, "\n\n")
    end
  end

  defp prepend_system_prompt(prompt, nil), do: prompt

  defp prepend_system_prompt(prompt, system),
    do: "<system>\n#{system}\n</system>\n\n#{prompt}"

  defp codex_usage_to_claude(%{} = usage) do
    %{
      base_usage()
      | input_tokens: usage["input_tokens"] || usage["inputTokens"] || 0,
        cache_read_input_tokens:
          usage["cached_input_tokens"] || usage["cachedInputTokens"] || 0,
        output_tokens: usage["output_tokens"] || usage["outputTokens"] || 0
    }
  end

  defp codex_usage_to_claude(_), do: base_usage()

  defp base_usage do
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

  defp format_mcp_result(%{"error" => %{"message" => msg}}) when is_binary(msg), do: msg

  defp format_mcp_result(%{"result" => %{"content" => content}}) when is_list(content) do
    Enum.map_join(content, "\n", fn
      %{"text" => text} when is_binary(text) -> text
      other -> inspect(other)
    end)
  end

  defp format_mcp_result(_), do: ""

  # --- JSON-RPC I/O ---

  defp send_rpc(state, method, params, tag) do
    id = state.rpc_counter + 1
    Port.command(state.port, encode_rpc(id, method, params))
    {id, %{state | rpc_counter: id, pending_rpc: Map.put(state.pending_rpc, id, tag)}}
  end

  defp encode_rpc(id, method, params) do
    Jason.encode!(%{"jsonrpc" => "2.0", "id" => id, "method" => method, "params" => params}) <>
      "\n"
  end

  # --- Port spawning ---

  defp open_codex_port(state) do
    path = state.session_options[:codex_path] || "codex"

    case resolve_executable(path) do
      nil ->
        {:error, {:codex_not_found, path}}

      exe ->
        port =
          Port.open({:spawn_executable, exe}, [
            {:args, ["app-server", "--transport", "stdio"]},
            :binary,
            :exit_status,
            :stderr_to_stdout
          ])

        {:ok, port}
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

  # --- Test seams ---

  @doc false
  def __dispatch__(json, state), do: dispatch(json, state)

  @doc false
  def __new_state__(session, opts \\ []) do
    %__MODULE__{
      session: session,
      session_options: opts,
      pending_system_prompt: build_system_prompt(opts),
      thread_id: opts[:thread_id],
      current_request: opts[:current_request],
      turn_started_at: System.monotonic_time(:millisecond),
      num_turns: opts[:num_turns] || 1,
      status: :ready
    }
  end
end
