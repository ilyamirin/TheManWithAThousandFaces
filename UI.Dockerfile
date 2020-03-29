FROM nginx:alpine

EXPOSE 80

COPY src/ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY src/ui/build /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]
