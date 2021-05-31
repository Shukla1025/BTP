const http = require('http');
const fs = require('fs');
const path = require('path');
const stream = require('stream');

const hostname = '';
const port = 3000;

const server = http.createServer((req, res) => {
	
	if (req.method == 'GET') {
		var filePath = path.resolve('.' + req.url.replace(/%20/g, ' ').replace(/%7B/g, '{').replace(/%7D/g, '}'));
		const fileExt = path.extname(filePath);
		try {
			const stat = fs.statSync(filePath)
		    const fileSize = stat.size
			res.statusCode = 200;
			res.setHeader('Content-Length', fileSize);
			res.setHeader('Content-Type', `image/${fileExt.replace(/./, '')}`);
			res.setHeader('Access-Control-Allow-Origin', '*');
			const file = fs.createReadStream(filePath);
			file.pipe(res);
		}
		catch (err) {
			res.statusCode = 403;
			res.setHeader('Content-Type', 'text/html');
			res.end('Could not stream image. Possible error in filename');
		}
	}
	else if (req.method == 'POST') {
	
		var filePath = path.resolve('.' + req.url.replace(/%20/g, ' ') + '/' + req.headers.name);
		var writeStream = fs.createWriteStream(filePath);
		console.log('Receiving file');

		req.pipe(writeStream);
		var spawn = require("child_process").spawn;

		var process = spawn('python',["./BTP_final.py",
                            filePath,
                            req.headers.algo] );


		process.stdout.on('data', function(data) {
			res.writeHead(200, {'Content-Type': 'text/html'});
        	res.end(data.toString());
    	} )

    	process.on('close', (code) => {
			console.log(`child process close all stdio with code ${code}`);
			});

		writeStream.on('error', (err) => {
		    console.log(err);
		});
	}
});

server.listen(port, hostname, () => {
	console.log(`Server running at http://${hostname}:${port}`)
});