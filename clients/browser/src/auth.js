import axios from 'axios';
import { ref } from 'vue';

export const USER_TOKEN_KEY = 'sism_t';

const token = window.localStorage.getItem(USER_TOKEN_KEY);
if (token) axios.defaults.headers.common.Authorization = `Bearer ${token}`;

export const isLoggedIn = ref(Boolean(token));

const LOGIN_USER = `
  mutation ($username: String!, $password: String!) {
    tokenAuth(username: $username, password: $password) {
      token
    }
  }
`;

export const login = async (username, password) => {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: LOGIN_USER,
    variables: { username, password },
  });
  if (resp.data.errors) throw new Error(resp.data.errors[0].message);

  const accessToken = resp.data.data.tokenAuth.token;
  axios.defaults.headers.common.Authorization = `Bearer ${accessToken}`;
  window.localStorage.setItem(USER_TOKEN_KEY, accessToken);
  isLoggedIn.value = true;
};

export const logout = async () => {
  window.localStorage.removeItem(USER_TOKEN_KEY);
  delete axios.defaults.headers.common.Authorization;
  isLoggedIn.value = false;
};
