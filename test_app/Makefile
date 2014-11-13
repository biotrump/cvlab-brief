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

CC=g++

SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

# Only enable -msse4.2 on CPUs supporting the POPCNT instruction
CFLAGS = -Wall -DNDEBUG -O3 -march=nocona #-msse4.2
#CFLAGS = -Wall -DDEBUG -g -O0 -fno-inline-functions
LDFLAGS = -Wl

# BRIEF
CFLAGS += -I../brief/include
LDFLAGS += -L../brief/lib -lbrief

# OpenCV
CFLAGS += `pkg-config opencv --cflags`
LDFLAGS += `pkg-config opencv --libs`

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) -g -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE) matches.png
