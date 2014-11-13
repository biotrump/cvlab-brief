#  Copyright 2010 Computer Vision Lab,
#  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
#  All rights reserved.
#
#  Author: Michael Calonder (http://cvlab.epfl.ch/~calonder)
#  Version: 1.0
#
#  This file is part of the BRIEF DEMO software.
#
#  BRIEF DEMO is free software; you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation; either version 2 of the License, or (at your option) any later
#  version.
#
#  BRIEF DEMO is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#  PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with
#  BRIEF DEMO; if not, write to the Free Software Foundation, Inc., 51 Franklin
#  Street, Fifth Floor, Boston, MA 02110-1301, USA

CXX=g++

SRCPATH = src
SOURCES = src/utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)
LIBRARY = libbrief

CFLAGS  = -c -fpic $(DBG_FLAGS) -Wall -DNDEBUG -Wall -O3 -march=nocona -msse2
LDFLAGS = -Wl

CFLAGS += -Iinclude

# OpenCV
CFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

all: $(SOURCES) $(LIBRARY).so

$(LIBRARY).so: $(OBJECTS)
	$(CXX) -shared $(LDFLAGS) $(OBJECTS) -o lib/$(LIBRARY).so

.cpp.o:
	$(CXX) $(CFLAGS) -c $(SOURCES) -o $(SOURCES:.cpp=.o)

clean:
	rm -f lib/$(LIBRARY).so $(OBJECTS)
