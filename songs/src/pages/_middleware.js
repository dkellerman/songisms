import { NextResponse } from 'next/server';

const PUBLIC_PATHS = ['/login'];

export async function middleware(req) {
  const loggedIn = !!req.cookies['sism2_u'];

  if (!PUBLIC_PATHS.includes(req.nextUrl.pathname) && !loggedIn) {
    return redirect(req, '/login');
  }
}

function redirect(req, to) {
  const url = req.nextUrl.clone();
  url.pathname = to;
  return NextResponse.redirect(url);
}
