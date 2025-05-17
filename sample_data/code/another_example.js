// search_project/sample_data/code/another_example.js

/**
 * Greets a user by name. Defaults to "Guest" if name is not provided or invalid.
 * Logs an error to the console if the name parameter is problematic.
 * @param {string} [name] - The name of the user to greet. Optional.
 * @returns {string} A greeting message.
 */
function greetUser(name) {
  if (!name || typeof name !== 'string' || name.trim() === '') {
    console.error("Error in greetUser: Name parameter is missing, not a string, or empty. Using default 'Guest'.");
    return "Hello, Guest!";
  }
  // Sanitize name slightly for display (very basic example)
  const sanitizedName = name.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `Hello, ${sanitizedName}! Welcome to our advanced application.`;
}

/**
 * Asynchronously fetches data from a specified URL using the Fetch API.
 * Handles potential network errors and non-OK HTTP responses robustly.
 * @param {string} url - The URL to fetch data from. Must be a valid URL string.
 * @returns {Promise<Object|null>} A promise that resolves to the JSON parsed data, 
 * or null if an error occurs during fetching or parsing.
 */
async function fetchData(url) {
  if (!url || typeof url !== 'string') {
    console.error("FetchData Error: URL parameter is invalid or missing.");
    return null; // Or throw new TypeError("URL must be a non-empty string");
  }
  try {
    const response = await fetch(url, {
      method: 'GET', // Explicitly state method, though GET is default
      headers: {
        'Accept': 'application/json', // Specify we expect JSON
      }
    });

    if (!response.ok) {
      // Attempt to get more error info from response if possible
      let errorBody = null;
      try {
        errorBody = await response.text(); // Or response.json() if server sends structured errors
      } catch (parseError) {
        // Ignore if response body can't be parsed
      }
      const errorMessage = `HTTP error! Status: ${response.status} (${response.statusText}) for URL: ${url}. ` +
                           (errorBody ? `Response body: ${errorBody.substring(0,100)}...` : 'No response body or unparsable.');
      throw new Error(errorMessage);
    }

    // Check content type before parsing, though fetch API might handle some of this
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.indexOf("application/json") === -1) {
        throw new Error(`Expected JSON response but got ${contentType} from ${url}`);
    }
    
    const data = await response.json();
    return data;

  } catch (error) { // Catches network errors and errors thrown above
    console.error('Failed to fetch or process data:', error.message); // Log the specific error message
    // Example of updating UI (if in a browser context)
    // if (typeof document !== 'undefined' && document.getElementById('fetch-error-display')) {
    //   document.getElementById('fetch-error-display').textContent = 'Could not load data. Please try again later.';
    // }
    return null; 
  }
}

// Example of a self-executing async function to demonstrate fetchData
// (async () => {
//   console.log(greetUser("Alice Wonderland"));
//   console.log(greetUser()); // Test greetUser error case

//   // This URL will likely fail, demonstrating the error handling of fetchData
//   const MOCK_API_URL = 'https://jsonplaceholder.typicode.com/todos/1'; // A working mock API
//   // const FAILING_API_URL = 'https://api.example.com/nonexistentdata';
  
//   console.log(`\nFetching data from: ${MOCK_API_URL}`);
//   const fetchedData = await fetchData(MOCK_API_URL);
//   if (fetchedData) {
//     console.log('Successfully fetched data:', fetchedData);
//   } else {
//     console.log('No data was fetched or an error occurred.');
//   }
// })();

// Example of a common JavaScript error scenario for search indexing:
// const undefinedObject = undefined;
// console.log(undefinedObject.someProperty); // This would cause: TypeError: Cannot read properties of undefined (reading 'someProperty')