FROM nginx:alpine

# Init system
RUN apk add --update npm

WORKDIR /home/app

# Resolve app dependecies
COPY src/ui/package.json src/ui/
RUN npm install --prefix src/ui/

# Copy app sources
COPY src/ui/src src/ui/src
COPY src/ui/public src/ui/public
COPY src/ui/nginx.conf /etc/nginx/conf.d/default.conf

# Build app
RUN npm run build --prefix src/ui/
RUN mv src/ui/build /usr/share/nginx/html

EXPOSE 80

ENTRYPOINT ["nginx", "-g", "daemon off;"]
