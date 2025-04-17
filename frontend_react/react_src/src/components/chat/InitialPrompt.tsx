import { useState } from "react";
import { PromptInput } from "@/components/ui-custom/prompt-input";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPaperPlane } from "@fortawesome/free-solid-svg-icons/faPaperPlane";
import chatMidnight from "@/assets/chat-midnight.svg";
import { usePostMessage, useFetchAllChats } from "@/api-state/chat-messages/hooks";
import { useAppState } from "@/state/hooks";

export const InitialPrompt = ({ chatId }: { chatId?: string }) => {
  const { enableChartGeneration, enableBusinessInsights, dataSource: globalDataSource } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { mutate } = usePostMessage();
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
    <div className="flex-1 flex flex-col p-4">
      <div className="flex flex-col flex-1 items-center justify-center">
        <div className="w-[400px] flex flex-col flex-1 items-center justify-center">
          <img src={chatMidnight} alt="" />
          <h4 className="mb-2 mt-4">
            <strong className=" text-center font-semibold">
              Type a question about your dataset
            </strong>
          </h4>
          <p className="text-center mb-10">
            Ask specific questions about your datasets to get insights, generate visualizations, 
            and discover patterns. Include column names and the kind of analysis 
            you're looking for to get more accurate results.
          </p>
          <PromptInput
            icon={FontAwesomeIcon}
            iconProps={{
              icon: faPaperPlane,
              behavior: "append",
              onClick: sendMessage,
            }}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                sendMessage();
              }
            }}
            value={message}
            placeholder="Ask another question about your datasets."
          />
        </div>
      </div>
    </div>
  );
};
