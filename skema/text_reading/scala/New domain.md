# How to adapt SKEMA Text Reading to a new domain

### 1. Obtain the new domain specific MIRA DKG

The first necessary step is to download the DKG in its `json` format using CURL.
Once this file has been retrieved, add it into the `src/main/resources`

Current DKGs bundled with the system are:
- `src/main/resources/mira_dkg_epi_pretty.json` for the _epidemiology_ domain.
- `src/main/resources/mira_dkg_sw_pretty.json` for the _space weather_ domain.

Any new json file with a domain specific DKG should be within the same directory.

### 2. [Optional] Add domain specific word embedding model

For the case of domain specific word embedding model, use the W2V file format and drop it into the resources' directory: `src/main/resources`.

### 3. Configure the application with the new domain grounding information

Edit the configuration file, located at `src/main/resources/application.conf` to add domain specific configuration. This configuration should be inside the `Grounding` section. Examples of existing domains are: `Epidemiology` and `EarthSciences`.

To create a new domain, name it and add the two minimum required parameters: `ontologyPath` and `apiEndpoint`.
- `ontologyPath` is the file name of the json MIRA DKG form step 1. Make sure to prepend a slash (see example below).
- `apiEndpoint` is the URI of MIRA's native grounding mechanism. Get the URL's endpoint from the MIRA team.
- `embeddingsModelPath`: is the file name of the W2V model with domain specific embeddings. Similarly to the ontology path, prepend the name with a slash `/`.

The following example represents a bare-bones SpaceWeather. It doesn't have any word embeddidng model specified, so it will fall back to general domain glove embeddings.

```
Grounding {
      SpaceWeather{
      // Params for MiraEmbeddings
      ontologyPath
      =
      /mira_dkg_sw_pretty.json
      // Params for MiraWebApi
      apiEndpoint
      =
      "http://34.230.33.149:8773/api/ground_list"
      // Space weather MIRA endpoint
      // Params for the manual grounder
    }
}
```

### 4. Activate the grounding domain
Once a new domain has been specified, set the `domain` key inside the `Grounding` section to the desired domain before running Skema TR.

```
Grounding {
    // Domain to use. By default is Epidemiology, can also be SpaceWeather or EarthSciences
    domain = Epidemiology
   ...
 } 
```