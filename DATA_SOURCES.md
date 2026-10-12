# Data sources

All evaluation documents come from **令和7年版 情報通信白書** (Ministry of Internal Affairs and Communications / 総務省, published July 2025). The PDFs are **not committed**; `rag_eval/data.py` downloads them from the official URLs and the eval output records their SHA-256.

| File (local name) | Official URL | Pages |
|---|---|---|
| `r07_summary01.pdf` | https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r07/summary/summary01.pdf | 12 (概要) |
| `r07_n1110000.pdf` | https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r07/pdf/n1110000.pdf | 9 (第Ⅰ部 第1章 第1節) |
| `r07_n1120000.pdf` | https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r07/pdf/n1120000.pdf | 12 (第Ⅰ部 第1章 第2節) |

## License (verified 2026-10-12 on the official pages)

From the official secondary-use notice for this edition, https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r07/pdf/r07riyou.pdf (「令和７年版情報通信白書の二次利用について」):

> 令和７年版情報通信白書（冊子版、PDF 版、HTML 版…）は、別添「自由な二次利用の対象外となる図表リスト（令和 7 年版情報通信白書関係）」に掲載されている図表及び第三者の出典が表示されている文章を除き、どなたでも自由に、複製・改変・頒布・公衆送信等の二次利用ができます。商用目的での二次利用も可能です。
> …「クリエイティブ・コモンズ・ライセンス表示 4.0 国際」に基づき、出典の表示を条件として、自由な二次利用を許諾します。
> 令和７年版情報通信白書 by 総務省 is licensed under a Creative Commons 表示 4.0 国際 License. https://creativecommons.org/licenses/by/4.0/deed.ja

The MIC site-wide terms (https://www.soumu.go.jp/menu_kyotsuu/policy/tyosaku.html) additionally state that site content is usable under 「公共データ利用規約（第1.0版）」 (PDL 1.0) unless otherwise noted.

**Attribution:** 出典：「令和7年版 情報通信白書」（総務省）(URLs above). The gold questions in `data/gold.jsonl` and the short evidence quotes in it were 加工して作成 (edited/derived) from these documents by the repo author; they are not an official MIC product. Figures listed in the MIC exclusion list and third-party-sourced text are not redistributed here; only short text spans are quoted as evidence.
