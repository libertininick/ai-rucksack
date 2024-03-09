# ai-rucksack

*A sack of AI tools for your model building adventures.*

This is a set of tools and models I've curated from several projects over the years. I found
myself recreating this functionality over and over again with slight variations. Now this
collection is bundled together in a rucksack you can carry from project to project.

## Table of contents

- [Project Setup](#project-setup)
- [Running Tests](#running-tests)

## Project Setup

### 0. Install `hatch`

`ai-rucksack` uses [`hatch`](https://hatch.pypa.io/latest/) for project management. You'll
need it installed (ideally in an isolated environment) before setting up `ai-rucksack`.

### 1. Clone the `ai-rucksack` repository

```sh
git clone git@github.com:libertininick/ai-rucksack.git
```

### 2. Create the default (virtual) environment

`hatch` will install `ai-rucksack` in development mode along with its development dependencies
inside of a virtual environment managed by `hatch`.

```sh
cd ai-rucksack
hatch env create
```

[Table of Contents](#table-of-contents)

## Running Tests

Run tests and coverage report across the full environment matrix using `hatch`

```sh
hatch run test:cov
```

[Table of Contents](#table-of-contents)
