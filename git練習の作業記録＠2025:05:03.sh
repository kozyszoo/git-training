# sshでgit-cloneしたときの作業記録
~/projects/git-training

## 鍵生成
ssh-keygen -t ed25519 -C "suzuki.hirotaka@gmail.com"

## ssh-agent(+macOSのキーチェーン)
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
- 1回Touch IDで認証すればOK！
- パスフレーズはいつもの

## 公開鍵をクリップボードにコピー
pbcopy < ~/.ssh/id_ed25519.pub
- GitHubで登録


# git操作練習
