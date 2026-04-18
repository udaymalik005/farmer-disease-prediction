mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
port = \$PORT
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#52b788"
backgroundColor = "#0d2818"
secondaryBackgroundColor = "#1a3a2a"
textColor = "#e8f5e9"
EOF
