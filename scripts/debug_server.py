import json
from http.server import HTTPServer, BaseHTTPRequestHandler


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200, content_type='text/plain'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_PUT(self):
        content_length = int(self.headers['Content-Length'])
        put_data = self.rfile.read(content_length)

        try:
            data = json.loads(put_data.decode('utf-8'))
            print(f"Received data: {data}")
            self._set_headers(200)
            self.wfile.write("Received successfully".encode('utf-8'))
        except Exception as e:
            print(f"Error occurred while receiving data: {e}")
            self._set_headers(500)
            self.wfile.write("Error".encode('utf-8'))


def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Server stopped.')


if __name__ == "__main__":
    run()
