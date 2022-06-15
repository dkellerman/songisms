import axios from 'axios';
import { defineStore } from 'pinia';

export const USER_TOKEN_KEY = 'sism_t';

const LOGIN_USER = `
  mutation ($username: String!, $password: String!) {
    tokenAuth(username: $username, password: $password) {
      token
    }
  }
`;

const token = window.localStorage.getItem(USER_TOKEN_KEY);
if (token) axios.defaults.headers.common.Authorization = `Bearer ${token}`;

export const useAuth = defineStore('auth', {
  state: () => ({
    isLoggedIn: Boolean(token),
  }),
  actions: {
    async login(username, password) {
      const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
      const resp = await axios.post(url, {
        query: LOGIN_USER,
        variables: { username, password },
      });
      if (resp.data.errors) throw new Error(resp.data.errors[0].message);

      const accessToken = resp.data.data.tokenAuth.token;
      axios.defaults.headers.common.Authorization = `Bearer ${accessToken}`;
      window.localStorage.setItem(USER_TOKEN_KEY, accessToken);
      this.isLoggedIn = true;
    },

    async logout() {
      window.localStorage.removeItem(USER_TOKEN_KEY);
      delete axios.defaults.headers.common.Authorization;
      this.isLoggedIn = false;
    },
  },
});
