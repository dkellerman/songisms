<template>
  <nav>
    <h1><a href="/">Songisms</a></h1>
  </nav>
  <main>
    <WritersComponent />
  </main>
</template>

<script>
import WritersComponent from './components/WritersComponent.vue'

export default {
  name: 'App',
  components: {
    WritersComponent
  }
}
</script>

<style lang="scss">
@import "../node_modules/papercss/dist/paper.min.css";
@import "../../shared/layout.scss";
html, body, #app {
  padding: 0;
  margin: 0;
}
main {
  padding: 20px;
}
</style>
