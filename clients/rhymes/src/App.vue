<script>
export default {
  name: 'App',
};
</script>

<template>
  <nav>
    <h1><router-link to="/">Song Rhymes</router-link></h1>
    <div class="links">
      <a href="https://bipium.com">Metronome</a>
    </div>
  </nav>
  <main>
    <router-view />
  </main>
  <footer>
    SongRhymes by
    <a target="_blank" rel="noopener noreferer" href="https://linkedin.com/in/david-kellerman">&nbsp;David Kellerman</a>
    <div class="links">
      &nbsp;&mdash;
      <a target="_blank" rel="noopener noreferrer" href="https://github.com/dkellerman/songisms">&nbsp;Source code</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://bipium.com">&nbsp;Metronome</a>
      &nbsp;&#183;
      <a target="_blank" rel="noopener noreferrer" href="https://open.spotify.com/artist/2fxGUIL1BUCzWwKqP1ykUi">&nbsp;Music</a>
    </div>
</footer>
</template>

<style lang="scss">
@import '../node_modules/milligram/dist/milligram.min.css';
@import '../../shared/layout.scss';
html,
body,
#app {
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
nav {
  padding: 15px 20px 15px 20px;
  h1 a {
    color: black;
    font-size: 32px;
  }
  .links a {
    font-size: 14px;
    display: none;
  }
}
main {
  padding: 20px;
  flex: 1;
}
footer {
  border: 1px solid #eee;
  .links {
    display: none;
  }
  a, a:visited {
    color: blue;
  }
}
@media screen and (min-width: 640px) {
  footer .links {
    display: initial;
  }
}
</style>
