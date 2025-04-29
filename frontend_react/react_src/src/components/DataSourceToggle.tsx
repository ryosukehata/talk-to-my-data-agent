import React from "react";
import { useAppState } from "@/state/hooks";
import { DATA_SOURCES } from "@/constants/dataSources";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";
import {
  useFetchAllChats,
  useUpdateChatDataSource,
} from "@/api-state/chat-messages/hooks";
import { useParams } from "react-router-dom";

/**
 * Toggle component for switching between database and catalog data sources
 */
export const DataSourceToggle: React.FC = () => {
  const { chatId } = useParams<{ chatId?: string }>();
  const { dataSource, setDataSource } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { mutate: updateChatDataSource } = useUpdateChatDataSource();

  const handleValueChange = (value: string) => {
    if (value) {
      if (chatId) {
        updateChatDataSource({ chatId, dataSource: value });
      }
      setDataSource(value);
    }
  };

  // Get current value - either from the chat or from global state
  const currentValue = chatId ? 
    (chats?.find((c) => c.id === chatId)?.data_source || dataSource) : 
    dataSource;

  return (
    <ToggleGroup
      type="single"
      value={currentValue}
      onValueChange={handleValueChange}
      className="bg-muted rounded-md p-1 shadow-sm"
    >
      <ToggleGroupItem value={DATA_SOURCES.DATABASE} className="text-sm">
        <div className="m-2">Database</div>
      </ToggleGroupItem>
      <ToggleGroupItem value={DATA_SOURCES.FILE} className="text-sm">
        <div className="m-2">Registry / File</div>
      </ToggleGroupItem>
    </ToggleGroup>
  );
};
