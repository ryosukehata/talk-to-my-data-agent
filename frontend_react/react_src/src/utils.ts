export function getBaseUrl() {
  let baseUrl = "http://localhost:8080";
  if (import.meta.env.MODE === "production") {
    // expected valid url https://app.datarobot.com/custom_applications/{appId}/api
    const fullUrl = window.location.origin + window.location.pathname;
    baseUrl = fullUrl.split("/").splice(0, 5).join("/");
  }
  return baseUrl;
}
