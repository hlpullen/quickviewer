# QuickViewer package

A package for interactively viewing medical image data.

## Installation

1. Install using [pip](https://pip.pypa.io/en/stable/):
   - Optionally create virtual environment, for example:
     ```
     mkdir test_area
     cd test_area
     virtualenv .
     source bin/activate
     ```
   - With [gitlab access via ssh keys](https://docs.gitlab.com/ee/ssh/):
     ```
     pip install git+ssh://git@codeshare.phy.cam.ac.uk:/hp346/quickviewer
     ```
   - Any updates can later be installed by running: 
     ```
     pip install --upgrade git+ssh://git@codeshare.phy.cam.ac.uk:/hp346/quickviewer
     ```

2. Install a local copy of the code using [git](https://git-scm.com) and [conda](https://docs.conda.io/):
   - Clone repository:
     - With [gitlab access via ssh keys](https://docs.gitlab.com/ee/ssh/):
       ```
       git clone git@codeshare.phy.cam.ac.uk:/hp346/quickviewer
       ```
     - With [gitlab access via token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html):
       ```
       git clone https://codeshare.phy.cam.ac.uk/hp346/quickviewer
       ```
       Note that this may not work with older versions of git.  It has
       been tested successfully with git version 2.21.1.
   - From top-level directory of cloned repository, create **quickviewer**
     environment:
     ```
     conda env create --file environment.yml
     ```
   - Activate **quickviewer** environment:
     ```
     conda activate quickviewer
     ```
