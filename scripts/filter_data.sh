# Trim text length across the frist 700 articles, to create a new smaller dataset
cat data/articles.json | jq -r '.[:700] | map(select(.text | length < 1000))' > data/articles_short.json
