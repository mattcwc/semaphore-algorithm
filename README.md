# Semaphore - Image Processing Algorithm

Semaphore is a system that monitors physical mailboxes for deliveries and notifies users when mail arrives. In addition, Semaphore also categorizes and counts the number of items that are currently inside the mailbox and displays the information to the user in an associated smartphone application.  

This repository contains the Image Processing Algorithm component.

Semaphore was created for the Electrical and Computer Engineering Capstone Design Symposium 2017 at the University of Waterloo.


## About Semaphore
See the [main project page](https://shlchoi.github.io/semaphore) for more information.

### Other Semaphore Repositories
* [Mailbox Device](https://github.com/shlchoi/semaphore-mailbox)
* [Web Server](https://github.com/shlchoi/semaphore-server)
* [Android Application](https://github.com/shlchoi/semaphore-android)
* [iOS Application](https://github.com/shlchoi/semaphore-ios)

## Installation Instructions

Run `pip install -r requirements.txt` from root folder

## Usage
`run_algorithm(img, empty, plot)`   

|Parameter|Required|Default|Type   |Description             |Example                  |
|:--------|:------:|:-----:|:-----:|:-----------------------|:------------------------|
|img      |Required|None   |string |Path to image file      |`D:/images/image.jpg`    |
|empty    |Required|None   |string |Path to empty image file|`D:/images/calibrate.jpg`|
|plot     |No      |`false`|boolean|Display plot		       	|`true`                   |



## Authors
* Samson Choi 	[Github](https://github.com/shlchoi)
* Matthew Chum 	[Github](https://github.com/mattcwc)
* Lawrence Choi	[Github](https://github.com/l2choi)
* Matthew Leung [Github](https://github.com/mshleung)


## Acknowledgments
* [Armin Ronacher](http://lucumr.pocoo.org/about/) - [Flask](http://flask.pocoo.org/)
* [Ozgur Vatansever](https://github.com/ozgur) - [Python-firebase](http://ozgur.github.io/python-firebase/)
* OpenCV Dev Team - [opencv-python](http://docs.opencv.org/3.0-beta/)
* NumPy Developers - [NumPy](http://www.numpy.org/)
* PyData Development Team - [pandas](http://pandas.pydata.org/)
* Scikit-image Development Team - [scikit-image](http://scikit-image.org/)
* Matplotlib Development Team - [matplotlib](http://matplotlib.org/)
* SciPy Developers - [SciPy](https://www.scipy.org/)

## License

Distributed under the GNU GPLv3 license. See [LICENSE](https://github.com/shlchoi/semaphore-android/blob/master/LICENSE) for more information.

Libraries are used under the [BSD License](https://opensource.org/licenses/BSD-3-Clause), the [MIT License](https://opensource.org/licenses/MIT) and the [Python Software Foundation License](https://docs.python.org/3/license.html).

### BSD License
* [Flask](http://flask.pocoo.org/)
* [opencv-python](http://docs.opencv.org/3.0-beta/index.html)
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [scikit-image](http://scikit-image.org/)
* [SciPy](https://www.scipy.org/)

### MIT License
* [Python-firebase](http://ozgur.github.io/python-firebase/)

### Python Software Foundation License
* [matplotlib](http://matplotlib.org/)
