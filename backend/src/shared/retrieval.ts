import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { OpenAIEmbeddings } from '@langchain/openai';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { createClient } from '@supabase/supabase-js';
import { RunnableConfig } from '@langchain/core/runnables';
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import {
  BaseConfigurationAnnotation,
  ensureBaseConfiguration,
} from './configuration.js';

export async function makeSupabaseRetriever(
  configuration: typeof BaseConfigurationAnnotation.State,
): Promise<VectorStoreRetriever> {
  if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error(
      'SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables are not defined',
    );
  }
  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-small',
  });
  const supabaseClient = createClient(
    process.env.SUPABASE_URL ?? '',
    process.env.SUPABASE_SERVICE_ROLE_KEY ?? '',
  );
  const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: 'documents',
    queryName: 'match_documents',
  });
  return vectorStore.asRetriever({
    k: configuration.k,
    filter: configuration.filterKwargs,
  });
}

async function makeMemoryRetriever(configuration: any) {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  return vectorStore.asRetriever();
}

export async function makeRetriever(
  config: RunnableConfig,
): Promise<VectorStoreRetriever> {
  const configuration = ensureBaseConfiguration(config);
  switch (configuration.retrieverProvider) {
    case 'supabase':
      return makeSupabaseRetriever(configuration);
    case 'memory':
      return makeMemoryRetriever(configuration);
    default:
      throw new Error(
        `Unsupported retriever provider: ${configuration.retrieverProvider}`,
      );
  }
}
