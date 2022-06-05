import { useState, useEffect } from 'react';
import memoize from 'lodash/memoize';
import { ApolloClient, InMemoryCache, HttpLink, ApolloLink } from '@apollo/client';
import { persistCache, SessionStorageWrapper } from 'apollo3-cache-persist';
import { getUser } from './useAuth';

export const clientConfig = {
  accessToken: null,
};

export const getAPIClient = memoize(async ctx => {
  const cache = new InMemoryCache();

  if (process.env.NODE_ENV !== 'development' && typeof window !== 'undefined') {
    await persistCache({
      cache,
      storage: new SessionStorageWrapper(window.sessionStorage),
    });
  }

  const httpLink = new HttpLink({
    uri: `${process.env.NEXT_PUBLIC_SISM_API_BASE_URL}/graphql/`,
  });

  const authLink = new ApolloLink((operation, forward) => {
    const accessToken = clientConfig.accessToken ?? getUser(ctx)?.accessToken;

    if (accessToken) {
      operation.setContext({
        headers: {
          authorization: `Bearer ${accessToken}`,
        },
      });
    }
    return forward(operation);
  });

  const client = new ApolloClient({
    link: authLink.concat(httpLink),
    cache,
    credentials: 'include',
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
