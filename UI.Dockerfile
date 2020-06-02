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
ARG app_api_url
ENV REACT_APP_API_URL=$app_api_url

RUN npm run build --prefix src/ui/
RUN rm -rf /usr/share/nginx/html
RUN mv src/ui/build /usr/share/nginx/html

EXPOSE 80

ENTRYPOINT ["nginx", "-g", "daemon off;"]
