#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <vector>
#include <bitset>
#include <cmath>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <cassert>
#include <filesystem>
#include <cstdlib>

using namespace std;

const float limit_max_abs = 0.9999999;
const int num_threads = 48;

const int KB = 1024;
const int MB = KB * 1024;
const int GB = MB * 1024;

struct ModelInfo {
    std::string model_path;
    int para_num;
    // 0:float16, 1:float32, 2:float64
    int precision;
    // 0:compression, 1:decompression
    int flg = 0;

    std::string model_name;
    float model_size;
    vector<double> duration_vec;
    double duration_avg;
    float throughput;
    float throughput_decmp;
    double duration_avg_decmp;
};

template <typename T>
bool dumpVectorToBinaryFile(const std::vector<T>& data, const std::string& filePath) {
    // Open the file for writing
    int fd = open(filePath.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        perror("open");
        return false;
    }

    // Calculate the size needed for the file
    off_t fileSize = data.size() * sizeof(T);

    // Set the file size using ftruncate
    if (ftruncate(fd, fileSize) == -1) {
        perror("ftruncate");
        close(fd);
        return false;
    }

    // Map the file into memory
    T* mappedData = (T*)mmap(NULL, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);
    if (mappedData == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return false;
    }

    // Copy the vector data to the mapped memory
    std::copy(data.begin(), data.end(), mappedData);

    // Unmap the memory
    if (munmap(mappedData, fileSize) == -1) {
        perror("munmap");
    }

    // Close the file
    close(fd);

    //std::cout << "Vector dumped to file: " << filePath << std::endl;

    return true;
}

void serializeBitsetArray(const std::vector<std::bitset<24>>& bitsetArray, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    for (const auto& bits : bitsetArray) {
        unsigned long value = bits.to_ulong();
        char bytes[3];
        bytes[0] = static_cast<char>((value >> 16) & 0xFF); // high 8
        bytes[1] = static_cast<char>((value >> 8) & 0xFF);  // middle 8
        bytes[2] = static_cast<char>(value & 0xFF);         // low 8
        outFile.write(bytes, 3);
    }
    outFile.close();
}

void elf_func(float* data, size_t start, size_t end, string over_para_list_file, string over_position_list_file, string within_para_file, int index){
    int total = int(end - start);
    
    std::vector<float> over_para_list;
    std::vector<int> over_position_list;
    std::vector<std::bitset<24>> bitsetArray;
    
    //char* elf = new char[total*3];
    
    over_para_list.reserve(int(total*0.04));
    over_position_list.reserve(int(total*0.04));
    bitsetArray.reserve(total*24);

    int postion = 0;
    for (size_t i = start; i < end; ++i) {
        float weight = data[i];
        
        if (std::fabs(weight)<limit_max_abs){
            std::bitset<1> flg =0;
            if(weight<0){
                flg=1;
            }
            weight = std::fabs(weight);
            weight+=1;
            uint32_t intRepresentation;
            std::memcpy(&intRepresentation, &weight, sizeof(float));
            /* 
            uint32_t last23Bits = static_cast<uint32_t>(weight) & 0x7FFFFF;
            uint32_t last24Bits = (last23Bits << 1) | flg.to_ulong();
            elf[cnt++] = (char)((last24Bits >> 16) & 0xFF);
            elf[cnt++] = (char)((last24Bits >> 8) & 0xFF);
            elf[cnt++] = (char)(last24Bits & 0xFF);
            */
            uint32_t last23Bits = intRepresentation & 0x7FFFFF;
            std::bitset<24> bits((last23Bits << 1) | flg.to_ulong());
            bitsetArray.push_back(bits);
	    /*
            unsigned int val = bits.to_uint();
            char bytes0 = static_cast<char>((val >> 16) & 0xFF); // high 8
            char bytes1 = static_cast<char>((val >> 8) & 0xFF);  // middle 8
            char bytes2 = static_cast<char>(val & 0xFF);         // low 8
            elf[cnt++] = bytes0;
            elf[cnt++] = bytes1;
            elf[cnt++] = bytes2;
            */
        }
        else{
            over_para_list.push_back(weight);
            over_position_list.push_back(postion);
        }
        postion++;
    }
    
    over_para_list_file     = over_para_list_file + to_string(index)+".bin";
    over_position_list_file = over_position_list_file + to_string(index)+".bin";
    within_para_file        = within_para_file + to_string(index)+".bin";
    if (!over_para_list.empty()) {
        dumpVectorToBinaryFile(over_para_list, over_para_list_file);
        dumpVectorToBinaryFile(over_position_list, over_position_list_file);
    }
    serializeBitsetArray(bitsetArray, within_para_file);
}

std::vector<std::bitset<24>> deserializeBitsetArray(const std::string& filename) {
    std::vector<std::bitset<24>> bitsetArray;
    std::ifstream inFile(filename, std::ios::binary);
    char bytes[3];

    while (!inFile.eof()) {
        inFile.read(bytes, 3);

        if (inFile.gcount() < 3) break;

        unsigned long value = static_cast<unsigned char>(bytes[0]) << 16 |
                              static_cast<unsigned char>(bytes[1]) << 8 |
                              static_cast<unsigned char>(bytes[2]);
        bitsetArray.emplace_back(std::bitset<24>(value));
    }

    inFile.close();
    return bitsetArray;
}

void exp_decoding(std::vector<std::bitset<24>> &list,std::vector<float>& within_para_float){
    for(auto& bits:list){
        std::bitset<32> floatBits;
        floatBits[31] = 0; // sign bit 0
        // exponent 01111111
        floatBits[30] = 0; floatBits[29] = 1; floatBits[28] = 1; floatBits[27] = 1; floatBits[26] = 1; floatBits[25] = 1; floatBits[24] = 1; floatBits[23] = 1;
        // mantissa
        for(int i = 0; i < 23; ++i) {
            floatBits[i] = bits[i+1];
        }

        uint32_t floatAsInt = floatBits.to_ulong();
        float value = *reinterpret_cast<float*>(&floatAsInt);

        value -= 1;

        if (bits[0] == 1) {
            value = -value;
        }
        within_para_float.push_back(value);
    }
}

void exp_decoding_with_char(unsigned char* within_para_list, int within_data_len, std::vector<float>& within_para_float){
    for (int i = 0; i < within_data_len; i += 3){
        std::bitset<32> floatBits;
        floatBits[31] = 0; // sign 0
        // exponent
        floatBits[30] = 0; floatBits[29] = 1; floatBits[28] = 1; floatBits[27] = 1; floatBits[26] = 1; floatBits[25] = 1; floatBits[24] = 1; floatBits[23] = 1;
	// mantissa
	char high = within_para_list[i];
	char middle = within_para_list[i+1];
	char low = within_para_list[i+2];
	int cnt = 22;
        for (int j = 0; j < 8; j++) {
	    floatBits[cnt--] = (high >> (7-j)) & 1;
	} 
	for (int j = 0; j < 8; j++) {
            floatBits[cnt--] = (middle >> (7-j)) & 1;
        }
	for (int j = 0; j < 7; j++) {
	    floatBits[cnt--] = (low >> (7-j)) & 1;
	}
	assert(cnt==-1);
	uint32_t floatAsInt = floatBits.to_ulong();
        float value = *reinterpret_cast<float*>(&floatAsInt);
        value -= 1;
        if (low & 1 == 1) {
            value = -value;
        }
        within_para_float.push_back(value);
    }
}

void elf_func_decmp(string over_para_list_file, string over_position_list_file, string within_para_file, int index, vector<float>& thread_weights) {
    if (!std::filesystem::exists(over_para_list_file)) {
        //mmap read within_para_file
        int fd_within = open(within_para_file.c_str(), O_RDWR);
        if (fd_within == -1) {
            perror("open");
        }
        struct stat fileStat_within;
        if (fstat(fd_within, &fileStat_within) < 0) {
            perror("fstat");
            close(fd_within);
        }
        void* mapped_data_within = mmap(NULL, fileStat_within.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_within, 0);
        if (mapped_data_within == MAP_FAILED) {
            perror("mmap");
            close(fd_within);
        }
        unsigned char* within_para_list = static_cast<unsigned char*>(mapped_data_within);
        //cout << "deccmp: thread " <<  index << ", fileStat.st_size:" << fileStat_para.st_size << ", first:" << over_para_list[0] << endl;
        int within_data_len = fileStat_within.st_size;
        
        exp_decoding_with_char(within_para_list, within_data_len, thread_weights);
        
	// Unmap and close the file
        if (munmap(mapped_data_within, fileStat_within.st_size) == -1) {
            perror("munmap");
        }
        close(fd_within);

	/*
        std::vector<std::bitset<24>> within_para_list = deserializeBitsetArray(within_para_file);
        std::vector<float> within_para_float;
        exp_decoding(within_para_list, within_para_float);
	for (int i = 0; i < within_para_float.size(); i++) {
	    thread_weights.push_back(within_para_float[i]);
	}
	*/
    } else {
        int over_para_len = 0;
        int over_position_len = 0;
	//cout << "decmp: over_para_list_file:" << over_para_list_file << endl;
        int fd_para = open(over_para_list_file.c_str(), O_RDWR);
        if (fd_para == -1) {
            perror("open");
        }
        struct stat fileStat_para;
        if (fstat(fd_para, &fileStat_para) < 0) {
            perror("fstat");
            close(fd_para);
        }
        void* mapped_data_para = mmap(NULL, fileStat_para.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_para, 0);
        if (mapped_data_para == MAP_FAILED) {
            perror("mmap");
            close(fd_para);
        }
        float* over_para_list = static_cast<float*>(mapped_data_para);
        //cout << "deccmp: thread " <<  index << ", fileStat.st_size:" << fileStat_para.st_size << ", first:" << over_para_list[0] << endl;
        over_para_len = fileStat_para.st_size/4;
        
	//cout << "decmp: over_position_list_file:" << over_position_list_file << endl;
        int fd_pos = open(over_position_list_file.c_str(), O_RDWR);
        if (fd_pos == -1) {
            perror("open");
        }
        struct stat fileStat_pos;
        if (fstat(fd_pos, &fileStat_pos) < 0) {
            perror("fstat");
            close(fd_pos);
        }
        void* mapped_data_pos = mmap(NULL, fileStat_pos.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_pos, 0);
        if (mapped_data_pos == MAP_FAILED) {
            perror("mmap");
            close(fd_pos);
        }
        int* over_position_list = static_cast<int*>(mapped_data_pos);
        //cout << "deccmp: thread " <<  index << ", fileStat.st_size:" << fileStat_pos.st_size << ", first:" << over_position_list[0] << endl;
        over_position_len = fileStat_pos.st_size/4;
        
        //mmap read within_para_file
	int fd_within = open(within_para_file.c_str(), O_RDWR);
        if (fd_within == -1) {
            perror("open");
        }
        struct stat fileStat_within;
        if (fstat(fd_within, &fileStat_within) < 0) {
            perror("fstat");
            close(fd_within);
        }
        void* mapped_data_within = mmap(NULL, fileStat_within.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_within, 0);
        if (mapped_data_within == MAP_FAILED) {
            perror("mmap");
            close(fd_within);
        }
        unsigned char* within_para_list = static_cast<unsigned char*>(mapped_data_within);
        //cout << "deccmp: thread " <<  index << ", fileStat.st_size:" << fileStat_para.st_size << ", first:" << over_para_list[0] << endl;
        int within_data_len = fileStat_within.st_size;

	std::vector<float> within_para_float;
	within_para_float.reserve(within_data_len/3);
        exp_decoding_with_char(within_para_list, within_data_len, within_para_float);

        /* 
        std::vector<std::bitset<24>> within_para_list = deserializeBitsetArray(within_para_file);
        std::vector<float> within_para_float;
        exp_decoding(within_para_list, within_para_float);
        */
        //cout << "deccmp: thread " <<  index << ", within_para_float.size():" << within_para_float.size() << ", over_position_len:" << over_position_len << ", over_para_len:" << over_para_len << endl << endl;

        for (int pos_i = 0, over_i = 0, within_i = 0; pos_i < within_para_float.size() + over_position_len; ++pos_i) {
            if (over_i < over_position_len && pos_i == over_position_list[over_i]) {
                thread_weights.push_back(over_para_list[over_i]);
                ++over_i;
            } else {
                thread_weights.push_back(within_para_float[within_i]);
                ++within_i;
            }
        } 	

        // Unmap and close the file
        if (munmap(mapped_data_para, fileStat_para.st_size) == -1) {
            perror("munmap");
        }
        close(fd_para);

        // Unmap and close the file
        if (munmap(mapped_data_pos, fileStat_pos.st_size) == -1) {
            perror("munmap");
        }
        close(fd_pos);

	// Unmap and close the file
        if (munmap(mapped_data_within, fileStat_within.st_size) == -1) {
            perror("munmap");
        }
        close(fd_within);
    }
    
}


/*
void decompression(const std::vector<std::string>& files_for_decomp, std::vector<float>& weight){
    const std::string& over_para_list_file=files_for_decomp[0];
    const std::string& over_position_list_file=files_for_decomp[1];
    const std::string& within_para = files_for_decomp[2];


    std::vector<float> over_para = deserialize_vector<float>(over_para_list_file);
    std::vector<int> over_position = deserialize_vector<int>(over_position_list_file);
    std::vector<std::bitset<24>> within_para_list = deserializeBitsetArray(within_para);

    std::vector<float> within_para_float;

    exp_decoding(within_para_list, within_para_float);
    
    std::cout<<"finish exp decoding"<<std::endl;
    int pos_i =0;
    int over_i=0;
    int within_i=0;
    int last_pos=over_position.size()-1;
    for(int i = 0; i < within_para_float.size() + over_para.size(); ++i){
        if(pos_i != over_position[last_pos] && i == over_position[pos_i]){  // If it's time to insert over_para
            weight.push_back(over_para[over_i]);
            pos_i++;
            over_i++;
        }
        else{ // Else, insert within_para_float
            weight.push_back(within_para_float[within_i]);
            within_i++;
        }
    }
}
*/

int findSecondToLastSlash(const std::string& input) {
    int count = 0;
    for (int i = input.length() - 1; i >= 0; --i) {
        if (input[i] == '/') {
            count++;
            if (count == 2) {
                return i; // Return the position of the second-to-last slash
            }
        }
    }
    return -1;
}

bool createDirectory(const std::string& path) {
    std::filesystem::path dirPath(path);

    if (!std::filesystem::exists(dirPath)) {
        if (!std::filesystem::create_directory(dirPath)) {
            std::cerr << "Failed to create directory" << std::endl;
            return false;
        }
    } else if (!std::filesystem::is_directory(dirPath)) {
        std::cerr << "Path exists but is not a directory" << std::endl;
        return false;
    }

    return true;
}

int elf_cmp(ModelInfo& info) {
    double elapsed_total = 0;
    // Clear the page cache, dentries, and inodes (equivalent to "echo 3 > /proc/sys/vm/drop_caches" command)
    /*
    if (system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") != 0) {
        std::cerr << "Failed to clear caches." << std::endl;
        return 1;
    }
    */
    //std::cout << "Caches cleared successfully." << std::endl;

    // start record time
    auto start_read = std::chrono::high_resolution_clock::now();
    int fd = open(info.model_path.c_str(), O_RDWR);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct stat fileStat;
    if (fstat(fd, &fileStat) < 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    //std::cout << "fileStat.st_size:" << fileStat.st_size << ", static_cast<off_t>(info.para_num * sizeof(float)): " << static_cast<off_t>(info.para_num * sizeof(float)) << std::endl;
    if (fileStat.st_size != static_cast<off_t>(info.para_num * sizeof(float))) {
        std::cerr << "File size does not match expected size." << std::endl;
        close(fd);
        return 1;
    }
        
    void* mapped_data = mmap(NULL, fileStat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    float* data = static_cast<float*>(mapped_data);

    size_t secondSlashPos = findSecondToLastSlash(info.model_path);
    std::string file_folder = info.model_path.substr(0, secondSlashPos+1)+"exponential_dedup/";
    if (!createDirectory(file_folder))
        return 1;	

    string over_para_list_file     = file_folder+"over_para_list_file_";
    string over_position_list_file = file_folder+"over_position_list_file_";
    string within_para_file        = file_folder+"within_para_file_";

    // Calculate chunk size for each thread
    size_t chunk_size = info.para_num / num_threads;
    std::thread threads[num_threads];

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? info.para_num : (i + 1) * chunk_size;
        threads[i] = std::thread(elf_func, data, start, end, over_para_list_file, over_position_list_file, within_para_file, i);
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    // Unmap and close the file
    if (munmap(mapped_data, fileStat.st_size) == -1) {
        perror("munmap");
    }
    close(fd);

    auto end_read = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_read = end_read - start_read;
    //std::cout<<"Total Time: "<< elapsed_read.count() << " s." << " Throughput: " << fileStat.st_size/MB/elapsed_read.count() << " MB/s." << endl << endl;
    cout << "compression time: " << elapsed_read.count() << " s." << endl;
    return 0;
}

int elf_decmp(ModelInfo& info) {
    double elapsed_total = 0;
    // Clear the page cache, dentries, and inodes (equivalent to "echo 3 > /proc/sys/vm/drop_caches" command)
    /*
    if (system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") != 0) {
        std::cerr << "Failed to clear caches." << std::endl;
        return 1;
    }
    */
    //std::cout << "Caches cleared successfully." << std::endl;

    // start record time
    auto start_read = std::chrono::high_resolution_clock::now();
    // Create a vector to hold the modified data
    std::vector<float> decmp_vec(info.para_num);
    vector<vector<float>> threadData(num_threads);
    
    size_t chunk_size = info.para_num / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? info.para_num : (i + 1) * chunk_size;
        threadData[i].reserve(end-start);
    }

    std::string file_folder = info.model_path;	    
    //size_t secondSlashPos = findSecondToLastSlash(info.model_path);
    //std::string file_folder = info.model_path.substr(0, secondSlashPos+1)+"exponential_dedup/";

    string over_para_list_file     = file_folder+"over_para_list_file_";
    string over_position_list_file = file_folder+"over_position_list_file_";
    string within_para_file        = file_folder+"within_para_file_";
        
    std::thread threads[num_threads];	
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(elf_func_decmp, over_para_list_file+to_string(i)+".bin", over_position_list_file+to_string(i)+".bin", within_para_file+to_string(i)+".bin", i, std::ref(threadData[i]));
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
     }
    int cnt = 0;
    for (int i = 0; i < threadData.size(); i++){
        for (int j = 0; j < threadData[i].size(); j++) {
	    decmp_vec[cnt++] = threadData[i][j];
	}
    }
    //cout << "cnt:" << cnt << ", info.para_num:" << info.para_num << endl;
    string decmp_file_path = file_folder + "decmp.bin";
    dumpVectorToBinaryFile(decmp_vec, decmp_file_path);
    auto end_read = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_read = end_read - start_read;
    std::cout<<"decompression time: "<< elapsed_read.count() << " s." <<std::endl;
    return 0;
}

ModelInfo parse_args(int argc, char *argv[]){
    ModelInfo model_info;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i") {
            if (i + 1 < argc) {
                model_info.model_path = argv[++i];
            } else {
                std::cerr << "No input path specified after -i" << std::endl;
                exit(1);
            }
        } else if (arg == "-c") {
            model_info.flg = 1;
        } else if (arg == "-d") {
            model_info.flg = -1;
        } else if (arg == "-p") {
            if (i + 1 < argc) {
		std::string precision_str = argv[++i];
		if (precision_str == "f16")
	            model_info.precision = 0;
		else if (precision_str == "f32")
                    model_info.precision = 1;
		else if (precision_str == "f64")
                    model_info.precision = 2;
                else {
		    std::cerr << "Invalid precision. Choose from f16, f32, f64" << std::endl;
                    exit(1);
		}    
            } else {
                std::cerr << "No precision specified after -p" << std::endl;
                exit(1);
            }
        } else if (arg == "-n") {
            if (i + 1 < argc) {
                std::istringstream iss(argv[++i]);
                if (!(iss >> model_info.para_num)) {
                    std::cerr << "Invalid size. Size must be an integer." << std::endl;
                    exit(1);
                }
            } else {
                std::cerr << "No size specified after -size" << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
	    exit(1);
        }
    }

    return model_info;
}

int main(int argc, char *argv[]) {
    ModelInfo model_info = parse_args(argc, argv);
    if (model_info.flg == 1) {
        elf_cmp(model_info);
    } else if (model_info.flg == -1) {
        elf_decmp(model_info);
    } else {
        std::cerr << "please spicify the compression / decompression mode. -c for compression, -d for decompression. " <<  std::endl;
        exit(1);
    }
    
    return 0;
}


