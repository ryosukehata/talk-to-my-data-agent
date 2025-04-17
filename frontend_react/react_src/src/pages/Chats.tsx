import React, { Suspense, lazy } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@radix-ui/react-separator";
import { Button } from "@/components/ui/button";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTrash } from "@fortawesome/free-solid-svg-icons/faTrash";
import {
  useFetchAllMessages,
  useDeleteChat,
  useFetchAllChats,
} from "@/api-state/chat-messages/hooks";
import { InitialPrompt, UserMessage, UserPrompt } from "@/components/chat";
import { ROUTES } from "./routes";
import { Loading } from "@/components/ui-custom/loading";
import { RenameChatModal } from "@/components/RenameChatModal";
import { DataSourceToggle } from "@/components/DataSourceToggle";

// Lazy load ResponseMessage for better performance
const ResponseMessage = lazy(() =>
  import("../components/chat/ResponseMessage").then((module) => ({
    default: module.ResponseMessage,
  }))
);

const ComponentLoading = () => (
  <div className="p-4 text-sm">Loading component...</div>
);

export const Chats: React.FC = () => {
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();

  // API data hooks
  const { data: messages = [], status: messagesStatus } = useFetchAllMessages({
    chatId,
  });
  const { data: chats } = useFetchAllChats();
  const { mutate: deleteChat } = useDeleteChat();

  // Find the active chat based on chatId param
  const activeChat = chats
    ? chats.find((chat) => chat.id === chatId)
    : undefined;

  // Handler for deleting the current chat
  const handleDeleteChat = () => {
    if (activeChat?.id) {
      // Find another chat to navigate to after deletion
      const otherChats =
        chats?.filter((chat) => chat.id !== activeChat.id) || [];

      deleteChat({ chatId: activeChat.id });

      // If there are other chats, navigate to the most recently created one
      // Otherwise, go to the main chats page
      if (otherChats.length > 0) {
        // Sort by creation date descending (newest first) to get the most recent chat
        const sortedChats = [...otherChats].sort((a, b) => {
          const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
          const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
          return dateB - dateA; // Descending order (newest first)
        });

        // Navigate to the most recent chat
        navigate(`/chats/${sortedChats[0].id}`);
      } else {
        // If no other chats, go to the main chats page
        navigate(ROUTES.CHATS);
      }
    }
  };

  // Render chat messages
  const renderMessages = () => {
    if (!messages || messages.length === 0) {
      return (
        <Suspense fallback={<ComponentLoading />}>
          <InitialPrompt chatId={activeChat?.id} />
        </Suspense>
      );
    }

    return (
      <>
        <ScrollArea className="flex flex-1 flex-col overflow-y-hidden pr-2 pb-4">
          {messages?.map((message, index) => (
            <div key={index} className="flex flex-col">
              {message.role === "user" ? (
                <>
                  <UserMessage
                    id={index.toString()}
                    message={message.content}
                    timestamp={message.created_at}
                  />
                  {message.in_progress && (
                    <Suspense fallback={<ComponentLoading />}>
                      <ResponseMessage
                        id={index.toString()}
                        message={message}
                        isLoading={true}
                        chatId={chatId}
                      />
                    </Suspense>
                  )}
                </>
              ) : (
                <Suspense fallback={<ComponentLoading />}>
                  <ResponseMessage
                    id={index.toString()}
                    message={message}
                    isLoading={false}
                    chatId={chatId}
                  />
                </Suspense>
              )}
            </div>
          ))}
        </ScrollArea>
        <Suspense fallback={<ComponentLoading />}>
          <UserPrompt chatId={activeChat?.id} />
        </Suspense>
      </>
    );
  };

  // Render the header with chat title and actions
  const renderChatHeader = () => {
    if (!activeChat) return null;

    return (
      <>
        <h2 className="text-xl flex-1">
          <strong>{activeChat.name || "New Chat"}</strong>
          <RenameChatModal
            chatId={activeChat.id}
            currentName={activeChat.name}
          />
        </h2>
        <div>
          <DataSourceToggle />
        </div>
        <Button variant="ghost" onClick={handleDeleteChat}>
          <FontAwesomeIcon icon={faTrash} />
          <span className="ml-2">Delete chat</span>
        </Button>
      </>
    );
  };

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex justify-between items-center gap-2 h-9">
        {renderChatHeader()}
      </div>
      <Separator className="my-4 border-t" />

      {messagesStatus === "pending" && !messages?.length ? (
        <div className="flex items-center justify-center h-[calc(100vh-200px)]">
          <Loading />
        </div>
      ) : (
        renderMessages()
      )}
    </div>
  );
};
