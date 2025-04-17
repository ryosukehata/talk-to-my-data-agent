import React, { useEffect, useRef } from "react";
import { MessageHeader } from "./MessageHeader";
import { formatMessageDate } from "./utils";

interface UserMessageProps {
  id: string;
  date?: string;
  timestamp?: string;
  message?: string;
}

export const UserMessage: React.FC<UserMessageProps> = ({
  id,
  date,
  timestamp,
  message = "How many customers have a daytime call plan?",
}) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView(false);
  });

  // Use the formatted timestamp if available, otherwise fallback to date prop or default
  const displayDate = timestamp 
    ? formatMessageDate(timestamp) 
    : (date || "");

  return (
    <div
      key={id}
      className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
      ref={ref}
    >
      <MessageHeader name={"You"} date={displayDate} />
      <div className="self-stretch text-sm font-normal leading-tight">
        {message}
      </div>
    </div>
  );
};
