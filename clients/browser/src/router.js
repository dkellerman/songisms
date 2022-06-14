import { createRouter, createWebHistory } from 'vue-router';
import { isLoggedIn } from './auth';

const requireLogin = () => {
  if (!isLoggedIn.value) {
    return { path: '/login' };
  }
};

const routes = [
  { path: '/', redirect: '/writers' },
  {
    path: '/login',
    name: 'Login',
    component: () => import(/* webpackChunkName: "login" */ './components/LoginForm.vue'),
  },
  {
    path: '/writers',
    name: 'Writers',
    component: () => import(/* webpackChunkName: "writers" */ './components/WritersList.vue'),
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
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
