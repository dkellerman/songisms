import { createRouter, createWebHistory } from 'vue-router';
import { useAuth } from '@/stores/auth';
import { storeToRefs } from 'pinia';

const requireLogin = () => {
  const { isLoggedIn } = storeToRefs(useAuth());
  if (!isLoggedIn.value) {
    return { path: '/login' };
  }
};

const routes = [
  { path: '/', redirect: '/songs' },
  {
    path: '/login',
    name: 'Login',
    component: () => import(/* webpackChunkName: "login" */ './components/LoginForm.vue'),
  },
  {
    path: '/writers',
    name: 'Writers',
    component: () => import(/* webpackChunkName: "writers" */ './components/WritersList.vue'),
    beforeEnter: [requireLogin],
  },
  {
    path: '/writers/:id',
    name: 'WriterDetail',
    component: () => import(/* webpackChunkName: "writer" */ './components/WriterDetail.vue'),
    beforeEnter: [requireLogin],
  },
  {
    path: '/songs',
    name: 'Songs',
    component: () => import(/* webpackChunkName: "songs" */ './components/SongsList.vue'),
    beforeEnter: [requireLogin],
  },
  {
    path: '/songs/:id',
    name: 'SongDetail',
    component: () => import(/* webpackChunkName: "song" */ './components/SongDetail.vue'),
    beforeEnter: [requireLogin],
  },
  {
    path: '/ngrams',
    name: 'NGrams',
    component: () => import(/* webpackChunkName: "song" */ './components/NGrams.vue'),
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
