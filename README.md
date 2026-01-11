# multimodal-web-app（純前端展示）

期末專題展示頁（純前端）：手寫數字辨識（ONNX Runtime Web）+ 語音數字辨識（ONNX + 麥克風）+ 準確度統計（內建 MNIST 測試樣本）+ 專題說明頁。

## 開發/執行

在 `/home/wushengyong/multimodal-web-app`：

- 安裝依賴：`npm i`
- 開發：`npm run dev`
- 打包：`npm run build`
- 本機預覽 dist：`npm run preview`

注意：
- 麥克風錄音（`getUserMedia`）通常需要 `https://` 或 `http://localhost`，不要用 `file://` 直接打開 `dist/index.html`。

## 檔案輸出

- 成品靜態檔：`dist/`
- 靜態資源（模型/wasm/資料）：`public/`

## 雲端部署（靜態網站）

這個專案可以直接當「靜態網站」部署（不用後端）。

### GitHub Pages（推薦：自動部署）

已內建 GitHub Actions 工作流程：`multimodal-web-app/.github/workflows/deploy-pages.yml`

1. 把整個專案（不含 `node_modules/`、不含 `dist/`）推到 GitHub repo
2. 到 GitHub → Settings → Pages → Source 選 `GitHub Actions`
3. 之後每次 push 到 `main` 都會自動 build 並部署

### 其他平台

- Netlify / Vercel / Cloudflare Pages：Build Command=`npm run build`、Output=`dist`

提醒：
- 若要用語音辨識/麥克風，部署網址必須是 `https://`。

## 像 App 一樣使用

本專案包含 `public/manifest.webmanifest` + `public/sw.js`，在支援的瀏覽器上可用「加入主畫面/安裝」方式以近似 App 的方式使用。
