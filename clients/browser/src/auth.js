import axios from 'axios';
import { ref } from 'vue';

export const USER_KEY = 'sism3_u';

const user = ref(getUser());
if (user.value) {
  axios.defaults.headers.common.Authorization = `Bearer ${user.value.accessToken}`;
}

const LOGIN_USER = `
  mutation ($username: String!, $password: String!) {
    tokenAuth(username: $username, password: $password) {
      token
    }
  }
`;

const GET_CURRENT_USER = `
  query {
    user {
      username
    }
  }
`;

function getUser() {
  const user = window.localStorage.getItem(USER_KEY);
  return user && JSON.parse(user);
}

const login = async (username, password) => {
  const url = `${process.env.VUE_APP_SISM_API_BASE_URL}/graphql/`;
  const resp = await axios.post(url, {
    query: LOGIN_USER,
    variables: { username, password },
  });
  if (resp.data.errors) throw new Error(resp.data.errors[0].message);

  const accessToken = resp.data.data.tokenAuth.token;
  axios.defaults.headers.common.Authorization = `Bearer ${accessToken}`;

  const resp2 = await axios.post(url, {
    query: GET_CURRENT_USER,
  });
  if (resp2.data.errors) throw new Error(resp2.data.errors[0].message);

  const u = { ...resp2.data.data.user, accessToken };
  window.localStorage.setItem(USER_KEY, JSON.stringify(u));
  user.value = u;
  return u;
};

const logout = async () => {
  window.localStorage.removeItem(USER_KEY);
  delete axios.defaults.headers.common.Authorization;
};

const auth = {
  login,
  logout,
  user,
};

export default auth;
