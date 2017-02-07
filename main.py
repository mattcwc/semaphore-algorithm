from util.server_util import app, load_config

if __name__ == '__main__':
    app.config.update(load_config('config'))
    app.run(port=app.config['port'])
