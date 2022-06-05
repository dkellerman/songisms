import { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import styled from 'styled-components';
import Layout from "../components/Layout";

const LoginForm = styled.form`
  h2,
  input {
    margin-bottom: 20px;
  }
`;

export default function Login() {
  const [error, setError] = useState();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();

  async function onLogin(e) {
    e.preventDefault();
    if (!username || !password) {
      setError('Please fill in both username and password');
      return;
    }

    try {
      await login(username, password);
      window.location.href = '/';
    } catch (e) {
      console.error(e);
      setError(e.message);
    }
  }

  return (
    <Layout>
      <h2>Log in</h2>

      <LoginForm>
        {error && <div role="alert">{error}</div>}
        <fieldset>
          <label>Username:</label>
          <input type="text" onInput={e => setUsername(e.target.value?.trim())} />
          <label>Password:</label>
          <input type="password" onInput={e => setPassword(e.target.value?.trim())} />
        </fieldset>
        <button onClick={onLogin}>Log in</button>
      </LoginForm>
    </Layout>
  );
}
