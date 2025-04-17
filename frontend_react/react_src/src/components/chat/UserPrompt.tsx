import { useState } from "react";
import { PromptInput } from "@/components/ui-custom/prompt-input";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPaperPlane } from "@fortawesome/free-solid-svg-icons/faPaperPlane";
import { usePostMessage, useFetchAllChats } from "@/api-state/chat-messages/hooks";
import { useAppState } from "@/state";

export const UserPrompt = ({ chatId }: { chatId?: string }) => {
  const { mutate } = usePostMessage();
  const { enableChartGeneration, enableBusinessInsights, dataSource: globalDataSource } = useAppState();
  const { data: chats } = useFetchAllChats();
  const [message, setMessage] = useState("");
  
  // Find the active chat to get its data source setting
  const activeChat = chatId ? chats?.find(chat => chat.id === chatId) : undefined;
  const chatDataSource = activeChat?.data_source || globalDataSource;

  const sendMessage = () => {
    if (message.trim()) {
      mutate({ 
        message, 
        chatId,
        enableChartGeneration,
        enableBusinessInsights,
        dataSource: chatDataSource
      });
      setMessage("");
    }
  };

  return (
    <PromptInput
      icon={FontAwesomeIcon}
      iconProps={{
        icon: faPaperPlane,
        behavior: "append",
        onClick: sendMessage,
      }}
      placeholder="Ask another question about your datasets."
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          sendMessage();
        }
      }}
      onChange={(e) => setMessage(e.target.value)}
      value={message}
    />
  );
};
