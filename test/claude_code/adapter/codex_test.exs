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

  describe "adapter behaviour" do
    test "implements all required ClaudeCode.Adapter callbacks" do
      Code.ensure_loaded!(Codex)
      all_callbacks = ClaudeCode.Adapter.behaviour_info(:callbacks)
      optional_callbacks = ClaudeCode.Adapter.behaviour_info(:optional_callbacks)
      required_callbacks = all_callbacks -- optional_callbacks

      Enum.each(required_callbacks, fn {fun, arity} ->
        assert function_exported?(Codex, fun, arity),
               "Missing callback: #{fun}/#{arity}"
      end)
    end
  end

  describe "agentMessageDelta → PartialAssistantMessage text_delta" do
    setup :ready_state

    test "emits content_block_delta with text_delta on first delta", %{
      state: state,
      request_id: request_id
    } do
      notification = %{
        "method" => "agentMessageDelta",
        "params" => %{"itemId" => "msg_1", "delta" => "Hel"}
      }

      _state = Codex.__dispatch__(notification, state)

      assert_receive {:adapter_message, ^request_id,
                      %PartialAssistantMessage{
                        event: %{
                          type: :content_block_delta,
                          index: 0,
                          delta: %{type: :text_delta, text: "Hel"}
                        }
                      }}
    end

    test "reuses the same index across deltas for the same item", %{
      state: state,
      request_id: request_id
    } do
      state =
        Codex.__dispatch__(
          %{"method" => "agentMessageDelta", "params" => %{"itemId" => "m", "delta" => "a"}},
          state
        )

      _state =
        Codex.__dispatch__(
          %{"method" => "agentMessageDelta", "params" => %{"itemId" => "m", "delta" => "b"}},
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %PartialAssistantMessage{
                        event: %{index: 0, delta: %{type: :text_delta, text: "a"}}
                      }}

      assert_receive {:adapter_message, ^request_id,
                      %PartialAssistantMessage{
                        event: %{index: 0, delta: %{type: :text_delta, text: "b"}}
                      }}
    end
  end

  describe "reasoningTextDelta → thinking_delta" do
    setup :ready_state

    test "emits thinking_delta with a distinct index from text", %{
      state: state,
      request_id: request_id
    } do
      state =
        Codex.__dispatch__(
          %{"method" => "agentMessageDelta", "params" => %{"itemId" => "m1", "delta" => "hi"}},
          state
        )

      _state =
        Codex.__dispatch__(
          %{
            "method" => "reasoningTextDelta",
            "params" => %{"itemId" => "r1", "delta" => "because"}
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %PartialAssistantMessage{event: %{index: 0, delta: %{type: :text_delta}}}}

      assert_receive {:adapter_message, ^request_id,
                      %PartialAssistantMessage{
                        event: %{
                          index: 1,
                          delta: %{type: :thinking_delta, thinking: "because"}
                        }
                      }}
    end
  end

  describe "itemCompleted{agent_message} → AssistantMessage with TextBlock" do
    setup :ready_state

    test "emits a finalized AssistantMessage", %{state: state, request_id: request_id} do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "itemCompleted",
            "params" => %{
              "item" => %{"type" => "agent_message", "id" => "msg_1", "text" => "Hello world"}
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %AssistantMessage{
                        session_id: "thread-xyz",
                        message: %{content: [%TextBlock{type: :text, text: "Hello world"}]}
                      }}
    end
  end

  describe "itemCompleted{reasoning} → AssistantMessage with ThinkingBlock" do
    setup :ready_state

    test "emits ThinkingBlock with empty signature (codex has no signature)", %{
      state: state,
      request_id: request_id
    } do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "itemCompleted",
            "params" => %{
              "item" => %{"type" => "reasoning", "id" => "r1", "text" => "let me think"}
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %AssistantMessage{
                        message: %{
                          content: [
                            %ThinkingBlock{
                              type: :thinking,
                              thinking: "let me think",
                              signature: ""
                            }
                          ]
                        }
                      }}
    end
  end

  describe "itemCompleted{command_execution} → synthetic tool_use + tool_result pair" do
    setup :ready_state

    test "emits paired AssistantMessage (ToolUseBlock) and UserMessage (ToolResultBlock)", %{
      state: state,
      request_id: request_id
    } do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "itemCompleted",
            "params" => %{
              "item" => %{
                "type" => "command_execution",
                "id" => "cmd_42",
                "command" => "ls -la",
                "aggregated_output" => "total 0\n",
                "exit_code" => 0,
                "status" => "completed"
              }
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
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

      assert_receive {:adapter_message, ^request_id,
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

    test "marks the tool_result as error when exit_code is non-zero", %{
      state: state,
      request_id: request_id
    } do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "itemCompleted",
            "params" => %{
              "item" => %{
                "type" => "command_execution",
                "id" => "cmd_boom",
                "command" => "false",
                "aggregated_output" => "",
                "exit_code" => 1,
                "status" => "failed"
              }
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id, %AssistantMessage{}}

      assert_receive {:adapter_message, ^request_id,
                      %UserMessage{
                        message: %{content: [%ToolResultBlock{is_error: true}]}
                      }}
    end
  end

  describe "turnCompleted → ResultMessage" do
    setup :ready_state

    test "translates codex Usage into Claude Usage shape and nullifies current_request", %{
      state: state,
      request_id: request_id
    } do
      new_state =
        Codex.__dispatch__(
          %{
            "method" => "turnCompleted",
            "params" => %{
              "usage" => %{
                "input_tokens" => 10,
                "cached_input_tokens" => 3,
                "output_tokens" => 7
              }
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %ResultMessage{
                        subtype: :success,
                        is_error: false,
                        session_id: "thread-xyz",
                        num_turns: 1,
                        total_cost_usd: 0.0,
                        usage: %{
                          input_tokens: 10,
                          cache_read_input_tokens: 3,
                          output_tokens: 7
                        }
                      }}

      assert new_state.current_request == nil
    end

    test "accepts camelCase usage keys (app-server v2)", %{
      state: state,
      request_id: request_id
    } do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "turnCompleted",
            "params" => %{
              "usage" => %{
                "inputTokens" => 5,
                "cachedInputTokens" => 1,
                "outputTokens" => 4
              }
            }
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %ResultMessage{
                        usage: %{
                          input_tokens: 5,
                          cache_read_input_tokens: 1,
                          output_tokens: 4
                        }
                      }}
    end
  end

  describe "turnFailed → error ResultMessage" do
    setup :ready_state

    test "emits an error ResultMessage with the error message", %{
      state: state,
      request_id: request_id
    } do
      _state =
        Codex.__dispatch__(
          %{
            "method" => "turnFailed",
            "params" => %{"error" => %{"message" => "upstream 500"}}
          },
          state
        )

      assert_receive {:adapter_message, ^request_id,
                      %ResultMessage{
                        is_error: true,
                        subtype: :error_during_execution,
                        result: "upstream 500",
                        errors: ["upstream 500"]
                      }}
    end
  end

  describe "unknown methods" do
    setup :ready_state

    test "silently ignore unknown JSON-RPC methods", %{state: state} do
      state =
        Codex.__dispatch__(
          %{"method" => "someFutureNotification", "params" => %{"x" => 1}},
          state
        )

      refute_receive {:adapter_message, _, _}, 50
      assert %ClaudeCode.Adapter.Codex{} = state
    end

    test "silently ignore unknown item types in itemCompleted", %{state: state} do
      state =
        Codex.__dispatch__(
          %{
            "method" => "itemCompleted",
            "params" => %{"item" => %{"type" => "web_search", "id" => "w1"}}
          },
          state
        )

      refute_receive {:adapter_message, _, _}, 50
      assert %ClaudeCode.Adapter.Codex{} = state
    end
  end

  # ==========================================================================
  # Setup
  # ==========================================================================

  defp ready_state(_) do
    request_id = make_ref()

    state =
      Codex.__new_state__(self(),
        thread_id: "thread-xyz",
        current_request: request_id,
        model: "gpt-5-codex"
      )

    {:ok, state: state, request_id: request_id}
  end
end
