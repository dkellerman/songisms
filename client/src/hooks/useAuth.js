import { useEffect, useState, useCallback } from 'react';
import { parseCookies, setCookie, destroyCookie } from 'nookies';
import { useAPIClient, clientConfig } from './useAPIClient';
import { gql } from '@apollo/client';

export const USER_COOKIE = 'sism2_u';

const LOGIN_USER = gql`
  mutation ($username: String!, $password: String!) {
    tokenAuth(username: $username, password: $password) {
      token
    }
  }
`;

const GET_CURRENT_USER = gql`
  query {
    user {
      username
    }
  }
`;

export function getUser(ctx) {
  const cookies = parseCookies(ctx);
  const user = cookies[USER_COOKIE];
  return user && JSON.parse(user);
}

export function useAuth() {
  const [user, setUser] = useState();
  const client = useAPIClient();

  useEffect(() => {
    setUser(getUser());
  }, []);

  const login = useCallback(
    async (username, password) => {
      if (!client) return;

      let resp = await client.mutate({
        mutation: LOGIN_USER,
        variables: { username, password },
      });

      const accessToken = resp.data.tokenAuth.token;
      clientConfig.accessToken = accessToken;

      resp = await client.query({
        query: GET_CURRENT_USER,
      });

      const user = { ...resp.data.user, accessToken };
      setCookie(null, USER_COOKIE, JSON.stringify(user));
      setUser(user);
    },
    [client],
  );

  const logout = useCallback(async () => {
    destroyCookie(null, USER_COOKIE);
    clientConfig.accessToken = null;
    setUser(null);
  }, []);

  return { user, login, logout };
}
