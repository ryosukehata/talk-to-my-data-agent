import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { datasetKeys } from "./keys";
import { getDatasets, uploadDataset, deleteAllDatasets } from "./api-requests";
import { useState } from "react";
import { dictionaryKeys } from "../dictionaries/keys";
import { DictionaryTable } from "../dictionaries/types";

export const useFetchAllDatasets = ({ limit = 100 } = {}) => {
  const queryResult = useQuery({
    queryKey: datasetKeys.all,
    queryFn: ({ signal }) => getDatasets({ signal, limit }),
  });

  return queryResult;
};

export const useFileUploadMutation = ({ 
  onSuccess, 
  onError 
}: { 
  onSuccess: (data: unknown) => void; 
  onError: (error: Error) => void;
}) => {
  const [progress, setProgress] = useState(0);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ files, catalogIds }: { files: File[], catalogIds: string[] }) =>
      uploadDataset({
        files,
        catalogIds,
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const prg = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(prg);
          }
        },
      }),
    onMutate: async ({ files }) => {
      const previousDictionaries = queryClient.getQueryData<DictionaryTable[]>(dictionaryKeys.all) || [];
      
      const placeholderDictionaries: DictionaryTable[] = files.map(file => ({
        name: file.name,
        in_progress: true,
        column_descriptions: []
      }));
      
      queryClient.setQueryData<DictionaryTable[]>(
        dictionaryKeys.all,
        [...previousDictionaries, ...placeholderDictionaries]
      );
      
      return { previousDictionaries };
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
      onSuccess(data);
    },
    onError: (error, _, context) => {
      if (context?.previousDictionaries) {
        queryClient.setQueryData<DictionaryTable[]>(dictionaryKeys.all, context.previousDictionaries);
      }
      onError(error);
    },
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: datasetKeys.all }),
  });

  return { ...mutation, progress };
};

export const useDeleteAllDatasets = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () => deleteAllDatasets(),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: dictionaryKeys.all });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: dictionaryKeys.all });
    },
  });
  return mutation;
};
