package gr.aueb.delorean.chimp.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.nio.FloatBuffer;
import java.io.InputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.FileWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CommonConfigurationKeys;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.io.compress.brotli.BrotliCodec;
import org.apache.hadoop.hbase.io.compress.lz4.Lz4Codec;
import org.apache.hadoop.hbase.io.compress.xerial.SnappyCodec;
import org.apache.hadoop.hbase.io.compress.xz.LzmaCodec;
import org.apache.hadoop.hbase.io.compress.zstd.ZstdCodec;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.hadoop.io.compress.CompressionOutputStream;
import org.junit.jupiter.api.Test;

import fi.iki.yak.ts.compression.gorilla.ByteBufferBitInput;
import fi.iki.yak.ts.compression.gorilla.ByteBufferBitOutput;
import gr.aueb.delorean.chimp.Chimp32;
import gr.aueb.delorean.chimp.ChimpDecompressor32;
import gr.aueb.delorean.chimp.ChimpN32;
import gr.aueb.delorean.chimp.ChimpNDecompressor32;
import gr.aueb.delorean.chimp.Compressor32;
import gr.aueb.delorean.chimp.Decompressor32;
import gr.aueb.delorean.chimp.Value;



/**
 * These are generic tests to test that input matches the output after compression + decompression cycle, using
 * the value compression.
 *
 */
public class TestSinglePrecision {

	private static final int MINIMUM_TOTAL_BLOCKS = 50_000;
	private static String[] FILENAMES = {
            "/albert-base-v2.csv.gz",
            "/sentence-transformers_all-MiniLM-L6-v2.csv.gz"
	};



	@Test
    public void testChimp32() throws IOException {
	System.out.println("~~~~~ Chimp ~~~~~");
        for (String filename : FILENAMES) {
            TimeseriesFileReader timeseriesFileReader = new TimeseriesFileReader(getClass().getResourceAsStream(filename));
            long totalSize = 0;
            float totalBlocks = 0;
            double[] values;
            long encodingDuration = 0;
            long decodingDuration = 0;
             
            while ((values = timeseriesFileReader.nextBlock()) != null || totalBlocks < MINIMUM_TOTAL_BLOCKS) {
                if (values == null) {
                    timeseriesFileReader = new TimeseriesFileReader(getClass().getResourceAsStream(filename));
                    values = timeseriesFileReader.nextBlock();
                }
    
		/*
		System.out.println("length of values:" + values.length);
                for(double value : values) {
                    System.out.println(value);
                }
	        System.exit(0);	
                */

                Chimp32 compressor = new Chimp32();
                long start = System.nanoTime();
                for (double value : values) {
                    compressor.addValue((float) value);
                }
                compressor.close();
                encodingDuration += System.nanoTime() - start;
                totalSize += compressor.getSize();
                totalBlocks += 1;

                ChimpDecompressor32 d = new ChimpDecompressor32(compressor.getOut());
                for(Double value : values) {
                    start = System.nanoTime();
                    Value pair = d.readPair();
                    decodingDuration += System.nanoTime() - start;
                    assertEquals(value.floatValue(), pair.getFloatValue(), "Value did not match");
                }
                assertNull(d.readPair());

            }
            double chimp_bits = totalSize / (totalBlocks * TimeseriesFileReader.DEFAULT_BLOCK_SIZE);

	    System.out.println(String.format("Chimp32: %s - Bits/value: %.6f, Compression time per block: %.2f, Decompression time per block: %.2f", filename, totalSize / (totalBlocks * TimeseriesFileReader.DEFAULT_BLOCK_SIZE), encodingDuration / totalBlocks, decodingDuration / totalBlocks));
            String chimp_bits_file_path = "chimp_para_bits"+filename+".csv";
	    File file = new File(chimp_bits_file_path);
            file.getParentFile().mkdirs();  // Ensure directories exist
            try (FileWriter writer = new FileWriter(file)) {
                writer.write(Double.toString(chimp_bits));
	        System.out.println("Compression Ratio: "+Double.toString(32/chimp_bits)+".");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

	@Test
    public void testCorilla32() throws IOException {
	System.out.println("~~~~~ Gorilla ~~~~~");
        for (String filename : FILENAMES) {
            TimeseriesFileReader timeseriesFileReader = new TimeseriesFileReader(getClass().getResourceAsStream(filename));
            long totalSize = 0;
            float totalBlocks = 0;
            double[] values;
            long encodingDuration = 0;
            long decodingDuration = 0;
            while ((values = timeseriesFileReader.nextBlock()) != null || totalBlocks < MINIMUM_TOTAL_BLOCKS) {
                if (values == null) {
                    timeseriesFileReader = new TimeseriesFileReader(getClass().getResourceAsStream(filename));
                    values = timeseriesFileReader.nextBlock();
                }

                ByteBufferBitOutput output = new ByteBufferBitOutput();
                Compressor32 compressor = new Compressor32(output);
                long start = System.nanoTime();
                for (double value : values) {
                    compressor.addValue((float) value);
                }
                compressor.close();
                encodingDuration += System.nanoTime() - start;
                totalSize += compressor.getSize();
                totalBlocks += 1;

                ByteBuffer byteBuffer = output.getByteBuffer();
                byteBuffer.flip();
                ByteBufferBitInput input = new ByteBufferBitInput(byteBuffer);
                Decompressor32 d = new Decompressor32(input);
                for(Double value : values) {
                    start = System.nanoTime();
                    Value pair = d.readValue();
                    decodingDuration += System.nanoTime() - start;
                    assertEquals(value.floatValue(), pair.getFloatValue(), "Value did not match");
                }
                assertNull(d.readValue());

            }
	    double chimp_bits = totalSize / (totalBlocks * TimeseriesFileReader.DEFAULT_BLOCK_SIZE);

            System.out.println(String.format("Gorilla32: %s - Bits/value: %.6f, Compression time per block: %.2f, Decompression time per block: %.2f", filename, totalSize / (totalBlocks * TimeseriesFileReader.DEFAULT_BLOCK_SIZE), encodingDuration / totalBlocks, decodingDuration / totalBlocks));
            String chimp_bits_file_path = "gorilla_para_bits"+filename+".csv";
            File file = new File(chimp_bits_file_path);
            file.getParentFile().mkdirs();  // Ensure directories exist
            try (FileWriter writer = new FileWriter(file)) {
                writer.write(Double.toString(chimp_bits));
                System.out.println("Compression Ratio: "+Double.toString(32/chimp_bits)+".");
            } catch (IOException e) {
                e.printStackTrace();
            }
	}
    }

    public static float[] toFloatArray(byte[] byteArray){
        int times = Float.SIZE / Byte.SIZE;
        float[] floats = new float[byteArray.length / times];
        for(int i=0;i<floats.length;i++){
            floats[i] = ByteBuffer.wrap(byteArray, i*times, times).getFloat();
        }
        return floats;
    }


}
