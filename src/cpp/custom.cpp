#include <iostream>
#include <bitset>
#include <cstdint>

bool greater;

float reconstruct_fp16(uint8_t sign, uint8_t exp_lsb, uint8_t man_msb) {
    uint32_t new_bits = 0;

    std::cout << "Reconstruction" << std::endl;
    // Pad exponent LSBs into bits 23–25 (rest zero)
    std::cout << " new_bits = 0x" << std::bitset<32>(new_bits) << std::endl;
    
    if(greater)
    {
    	new_bits |= (((exp_lsb  & 0x07)) + 128 );
    }
    else
    {
    	new_bits |= (((exp_lsb  & 0x07)) - 127 );
    	std::cout << "exp(hex) - 127= 0x" << std::bitset<32>(new_bits) << std::endl;
    	uint32_t mask = 0xFFFFFF00; // ONly get 24 MSB
    	uint32_t msb_bits = ((~new_bits) & mask);
    	//msb_bits = ~msb_bits; // inverted MSB
std::cout << "exp(hex) MSB inverted= 0x" << std::bitset<32>(msb_bits) << std::endl;
    	uint32_t lsb_bits = new_bits &  0x00000007;
std::cout << "exp(hex) LSB = 0x" << std::bitset<32>(lsb_bits) << std::endl;
    	new_bits = msb_bits | lsb_bits;
    }
    
std::cout << "new_bits exp(hex) = 0x" << std::bitset<32>(new_bits) << std::endl;
     new_bits =new_bits << 23;
     std::cout << "new_bits exp(hex) post shift= 0x" << std::bitset<32>(new_bits) << std::endl;
   
	    
    // Pad mantissa MSBs into bits 19–22 (rest zero)
    new_bits |= (man_msb & 0x0F) << 19;

    std::cout << "new_bits man(hex) = 0x" << std::bitset<32>(new_bits) << std::endl;

    // Set sign bit at bit 31
    new_bits |= (sign & 0x1) << 31;
	    std::cout << "new_bits sign (hex) = 0x" << std::bitset<32>(new_bits) << std::endl;

    // Convert back to float
    float result = *reinterpret_cast<float*>(&new_bits);
    return result;
}

float simulate_fp16(float x) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&x);

    // Extract sign
    uint8_t sign = (bits >> 31) & 0x1;
	
    uint8_t exp_lsb = 0;
    // Extract 3 LSBs of exponent
    if(x>0)
    {
    	greater = true;
    	exp_lsb = exp_lsb - 127;
    }else {
    	greater = false;
    	exp_lsb = 127 - exp_lsb;
    }
    
//    uint8_t exp_lsb = exp_lsb -127;
    exp_lsb = ((bits >> 23) & 0xFF) & 0x07;

    // Extract 4 MSBs of mantissa
    uint8_t man_msb = (bits >> 19) & 0x0F;

    // Print extracted components
    std::cout << "Original float: " << x << std::endl;
    std::cout << "Sign = " << std::bitset<8>(sign) << std::endl;
    std::cout << "Exponent (3 LSBs) = " << std::bitset<8>(exp_lsb) << std::endl;
    std::cout << "Mantissa (4 MSBs) = " << std::bitset<8>(man_msb) << std::endl;

    // Reconstruct and print new float
    float reconstructed = reconstruct_fp16(sign, exp_lsb, man_msb);
    std::cout << "Reconstructed float: " << reconstructed << std::endl;

    return reconstructed;
}


int main() {
    float x;
    
    std::cout<< "Enter Number: ";
    std::cin>>x;
    
    float result = simulate_fp16(x);
    std::cout << "Final returned float: " << result << std::endl;
    return 0;
}

	
