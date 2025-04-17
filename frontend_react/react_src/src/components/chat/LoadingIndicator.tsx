import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCheck } from "@fortawesome/free-solid-svg-icons/faCheck";
import { faExclamationTriangle } from "@fortawesome/free-solid-svg-icons/faExclamationTriangle";
import loader from "@/assets/loader.svg";

interface LoadingIndicatorProps {
  isLoading?: boolean;
  hasError?: boolean;
}

export const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ 
  isLoading = true,
  hasError = false
}) => {
  if (hasError) {
    return (
      <FontAwesomeIcon 
        className="mr-2 w-4 h-4 text-destructive" 
        icon={faExclamationTriangle} 
        title="Error occurred during processing" 
      />
    );
  }
  
  return isLoading ? (
    <img src={loader} alt="processing" className="mr-2 w-4 h-4 animate-spin" />
  ) : (
    <FontAwesomeIcon className="mr-2 w-4 h-4 text-success" icon={faCheck} />
  );
};
