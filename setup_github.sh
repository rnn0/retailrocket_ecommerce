#!/usr/bin/env bash
# Conecta este repositório local ao seu GitHub e faz o primeiro push.
# Uso: ./setup_github.sh
set -e

echo "=== Setup do repositório GitHub ==="
echo

if [ -d ".git" ] && git remote get-url origin >/dev/null 2>&1; then
  echo "Este repositório já tem um remote 'origin' configurado:"
  git remote get-url origin
  read -p "Deseja substituir? (s/N) " resp
  if [[ "$resp" =~ ^[Ss]$ ]]; then
    git remote remove origin
  else
    echo "Cancelado."
    exit 0
  fi
fi

read -p "Cole a URL do repositório no GitHub (ex: https://github.com/seu-usuario/recsys-ecommerce.git): " REPO_URL

if [ -z "$REPO_URL" ]; then
  echo "URL vazia, abortando."
  exit 1
fi

git remote add origin "$REPO_URL"
echo
echo "Remote configurado. Enviando commits para o GitHub..."
echo "(Se pedir usuário/senha: use seu usuário do GitHub e, no lugar da senha,"
echo " um Personal Access Token gerado em https://github.com/settings/tokens"
echo " com o escopo 'repo'.)"
echo

git push -u origin main

echo
echo "Pronto! Repositório disponível em:"
echo "${REPO_URL%.git}"
