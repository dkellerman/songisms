import { useState, useEffect } from 'react';
import memoize from 'lodash/memoize';
import { ApolloClient, InMemoryCache, HttpLink, ApolloLink } from '@apollo/client';
import { persistCache, SessionStorageWrapper } from 'apollo3-cache-persist';

export const getAPIClient = memoize(async ctx => {
  const cache = new InMemoryCache();

  if (process.env.NODE_ENV !== 'development') {
    await persistCache({
      cache,
      storage: new SessionStorageWrapper(window.sessionStorage),
    });
  }

  const httpLink = new HttpLink({
    uri: `${process.env.NEXT_PUBLIC_SISM_API_BASE_URL}/graphql/`,
  });

  const client = new ApolloClient({
    link: httpLink,
    cache,
  });

  return client;
});

export function useAPIClient() {
  const [client, setClient] = useState();
  useEffect(() => {
    getAPIClient().then(setClient);
  }, []);
  return client;
}
