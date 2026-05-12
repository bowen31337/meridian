FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY src ./src
COPY . .
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD node -e "console.log('healthy')"
CMD ["node", "index.js"]
