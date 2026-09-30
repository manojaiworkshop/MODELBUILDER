import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lets the dev server (and its HMR websocket) be reached from other devices on the LAN,
  // e.g. opening http://192.168.1.6:3000 from a phone or another machine — by default Next
  // only trusts localhost and blocks everything else in dev mode.
  allowedDevOrigins: ["192.168.1.*","10.229.41.2"],
};

export default nextConfig;
