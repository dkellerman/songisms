import styled from 'styled-components';

export const StyledLayout = styled.div``;

export const Nav = styled.nav`
  padding: 10px 0px 10px 20px;
  background: aliceblue;
  display: flex;
  align-items: center;
  h1 {
    flex: 1;
    margin: 0;
    font-size: 36px;
    a,
    a:visited {
      border: 0;
    }
  }
  .links {
    white-space: nowrap;
    a,
    button {
      margin: 0 0 0 25px;
      color: black;
      font-size: medium;
    }
  }
`;

export const Main = styled.main`
  padding: 0 20px;
  h2 {
    font-size: 32px;
    margin-bottom: 10px;
  }
  fieldset {
    border: 0;
    padding: 0;
  }
  nav[aria-label='breadcrumbs'] {
    padding: 20px 0 10px 0;
  }
  a,
  a:visited {
    color: blue;
    border: 0;
  }
  div[role='alert'] {
    background: pink;
    color: darkred;
    border: red;
    padding: 8px;
    margin: 10px 0;
    width: fit-content;
  }
`;