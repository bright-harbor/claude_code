defmodule ClaudeCode.Adapter.CodexTest do
  use ExUnit.Case, async: true

  alias ClaudeCode.Adapter.Codex
  alias ClaudeCode.Content.TextBlock
  alias ClaudeCode.Content.ThinkingBlock
  alias ClaudeCode.Content.ToolResultBlock
  alias ClaudeCode.Content.ToolUseBlock
  alias ClaudeCode.Message.AssistantMessage
  alias ClaudeCode.Message.PartialAssistantMessage
  alias ClaudeCode.Message.ResultMessage
  alias ClaudeCode.Message.UserMessage

  setup do
    request_id = make_ref()

    state =
      Codex.__new_state__(self(),
        thread_id: "thread-xyz",
        current_request: request_id,
        model: "gpt-5-codex"
      )

    {:ok, state: state, request_id: request_id}
  end

  test "implements all required Adapter callbacks" do
    Code.ensure_loaded!(Codex)

    required =
      ClaudeCode.Adapter.behaviour_info(:callbacks) --
        ClaudeCode.Adapter.behaviour_info(:optional_callbacks)

    for {fun, arity} <- required do
      assert function_exported?(Codex, fun, arity), "Missing callback: #{fun}/#{arity}"
    end
  end

  describe "streaming deltas" do
    test "agentMessageDelta emits text_delta stream event",
         %{state: state, request_id: rid} do
      dispatch(state, "agentMessageDelta", %{"itemId" => "msg_1", "delta" => "Hel"})

      assert_receive {:adapter_message, ^rid,
                      %PartialAssistantMessage{
                        event: %{
                          type: :content_block_delta,
                          index: 0,
                          delta: %{type: :text_delta, text: "Hel"}
                        }
                      }}
    end

    test "successive deltas for the same item reuse the same index",
         %{state: state, request_id: rid} do
      state
      |> dispatch("agentMessageDelta", %{"itemId" => "m", "delta" => "a"})
      |> dispatch("agentMessageDelta", %{"itemId" => "m", "delta" => "b"})

      assert_receive {:adapter_message, ^rid,
                      %PartialAssistantMessage{event: %{index: 0, delta: %{text: "a"}}}}

      assert_receive {:adapter_message, ^rid,
                      %PartialAssistantMessage{event: %{index: 0, delta: %{text: "b"}}}}
    end

    test "reasoningTextDelta gets a distinct index from text and emits thinking_delta",
         %{state: state, request_id: rid} do
      state
      |> dispatch("agentMessageDelta", %{"itemId" => "m1", "delta" => "hi"})
      |> dispatch("reasoningTextDelta", %{"itemId" => "r1", "delta" => "because"})

      assert_receive {:adapter_message, ^rid,
                      %PartialAssistantMessage{event: %{index: 0, delta: %{type: :text_delta}}}}

      assert_receive {:adapter_message, ^rid,
                      %PartialAssistantMessage{
                        event: %{
                          index: 1,
                          delta: %{type: :thinking_delta, thinking: "because"}
                        }
                      }}
    end
  end

  describe "item completion" do
    test "agent_message item becomes AssistantMessage with TextBlock",
         %{state: state, request_id: rid} do
      dispatch_item(state, "agent_message", %{"id" => "msg_1", "text" => "Hello world"})

      assert_receive {:adapter_message, ^rid,
                      %AssistantMessage{
                        session_id: "thread-xyz",
                        message: %{content: [%TextBlock{text: "Hello world"}]}
                      }}
    end

    test "reasoning item becomes AssistantMessage with ThinkingBlock (empty signature)",
         %{state: state, request_id: rid} do
      dispatch_item(state, "reasoning", %{"id" => "r1", "text" => "let me think"})

      assert_receive {:adapter_message, ^rid,
                      %AssistantMessage{
                        message: %{
                          content: [
                            %ThinkingBlock{thinking: "let me think", signature: ""}
                          ]
                        }
                      }}
    end

    test "command_execution emits a paired ToolUseBlock + ToolResultBlock",
         %{state: state, request_id: rid} do
      dispatch_item(state, "command_execution", %{
        "id" => "cmd_42",
        "command" => "ls -la",
        "aggregated_output" => "total 0\n",
        "exit_code" => 0
      })

      assert_receive {:adapter_message, ^rid,
                      %AssistantMessage{
                        message: %{
                          content: [
                            %ToolUseBlock{
                              id: "codex_cmd_cmd_42",
                              name: "Bash",
                              input: %{"command" => "ls -la"}
                            }
                          ]
                        }
                      }}

      assert_receive {:adapter_message, ^rid,
                      %UserMessage{
                        message: %{
                          content: [
                            %ToolResultBlock{
                              tool_use_id: "codex_cmd_cmd_42",
                              content: "total 0\n",
                              is_error: false
                            }
                          ]
                        }
                      }}
    end

    test "non-zero exit_code marks the tool_result as error",
         %{state: state, request_id: rid} do
      dispatch_item(state, "command_execution", %{
        "id" => "boom",
        "command" => "false",
        "exit_code" => 1
      })

      assert_receive {:adapter_message, ^rid, %AssistantMessage{}}

      assert_receive {:adapter_message, ^rid,
                      %UserMessage{message: %{content: [%ToolResultBlock{is_error: true}]}}}
    end
  end

  describe "turn boundaries" do
    test "turnCompleted (snake_case) translates to success ResultMessage and clears request",
         %{state: state, request_id: rid} do
      new_state =
        dispatch(state, "turnCompleted", %{
          "usage" => %{
            "input_tokens" => 10,
            "cached_input_tokens" => 3,
            "output_tokens" => 7
          }
        })

      assert_receive {:adapter_message, ^rid,
                      %ResultMessage{
                        subtype: :success,
                        is_error: false,
                        session_id: "thread-xyz",
                        num_turns: 1,
                        total_cost_usd: 0.0,
                        usage: %{input_tokens: 10, cache_read_input_tokens: 3, output_tokens: 7}
                      }}

      assert new_state.current_request == nil
    end

    test "turnCompleted (camelCase) is also accepted",
         %{state: state, request_id: rid} do
      dispatch(state, "turnCompleted", %{
        "usage" => %{"inputTokens" => 5, "cachedInputTokens" => 1, "outputTokens" => 4}
      })

      assert_receive {:adapter_message, ^rid,
                      %ResultMessage{
                        usage: %{input_tokens: 5, cache_read_input_tokens: 1, output_tokens: 4}
                      }}
    end

    test "turnFailed emits an error ResultMessage with the message",
         %{state: state, request_id: rid} do
      dispatch(state, "turnFailed", %{"error" => %{"message" => "upstream 500"}})

      assert_receive {:adapter_message, ^rid,
                      %ResultMessage{
                        is_error: true,
                        subtype: :error_during_execution,
                        result: "upstream 500",
                        errors: ["upstream 500"]
                      }}
    end
  end

  describe "unknown methods" do
    test "unknown JSON-RPC method is silently ignored", %{state: state} do
      state = dispatch(state, "someFutureNotification", %{"x" => 1})
      refute_receive {:adapter_message, _, _}, 50
      assert %Codex{} = state
    end

    test "unknown itemCompleted item type is silently ignored", %{state: state} do
      state = dispatch_item(state, "web_search", %{"id" => "w1"})
      refute_receive {:adapter_message, _, _}, 50
      assert %Codex{} = state
    end
  end

  # --- Helpers ---

  defp dispatch(state, method, params),
    do: Codex.__dispatch__(%{"method" => method, "params" => params}, state)

  defp dispatch_item(state, type, fields),
    do: dispatch(state, "itemCompleted", %{"item" => Map.merge(%{"type" => type}, fields)})
end
