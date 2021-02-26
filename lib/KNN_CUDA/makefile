.PHONY : build reqs install clean
NINJA := $(shell command -v ninja 2> /dev/null)


build : reqs
	python3 setup.py bdist_wheel

reqs :
ifndef NINJA 
	cp ./ninja /usr/bin
endif 

install :
	pip3 install --upgrade dist/*.whl

clean :
	-rm -rf build dist/* *.egg-info
