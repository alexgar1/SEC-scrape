#include <iostream>
#include <string>
#include <regex>
#include <chrono>
#include <curl/curl.h>

// A callback function for libcurl to write the response data into a std::string
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    // Append data to the std::string provided via 'userp'
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main()
{
    // The SEC URL to query
    const std::string url = "https://www.sec.gov/Archives/edgar/data/1045810/";
    // Always use a descriptive User-Agent when making requests to sec.gov
    const std::string userAgent = "MyCompany (myemail@company.com)";

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize CURL handle
    CURL* curl = curl_easy_init();
    if(!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return 1;
    }

    std::string response;
    // Set CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    // Provide the write callback that stores the response in 'response'
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    // Set a custom User-Agent (very important for SEC)
    curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent.c_str());
    // Optional: if you're getting SSL verification issues, you might disable verify (not recommended for production)
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return 1;
    }

    // Clean up
    curl_easy_cleanup(curl);

    // Regex to capture the first link containing an <img src="/icons/folder.gif">
    // <a\s+href="([^"]+)"  => finds <a href="whatever">
    // [^>]*               => skip the rest of the anchor attributes
    // >\s*<img[^>]*src="/icons/folder.gif" => must contain an img with folder.gif
    std::regex pattern(R"(<a\s+href="([^"]+)"[^>]*>\s*<img[^>]*src="/icons/folder\.gif")");
    std::smatch match;

    bool found = std::regex_search(response, match, pattern);

    std::string link;
    if (found && match.size() > 1) {
        // Capture group #1 is the href
        link = match[1].str();
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    // Report execution time
    std::cout << "Execution time: " << diff.count() << " seconds" << std::endl;

    // Print the result
    if(!link.empty()) {
        std::cout << "First folder link found: " << link << std::endl;
    } else {
        std::cout << "No folder link found." << std::endl;
    }

    return 0;
}
