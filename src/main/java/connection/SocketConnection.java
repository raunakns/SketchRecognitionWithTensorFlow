package connection;

import java.io.*;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by david on 5/8/16.
 */
public class SocketConnection {

    /**
     * Declaration and Definition of Logger.
     */
    private static final Logger LOG = LoggerFactory.getLogger(SocketConnection.class);

    Socket pythonSocket;
    DataOutputStream out;
    DataInputStream in;

    public void createSocket(int portNumber) throws IOException {
        pythonSocket = new Socket("localhost", portNumber);
        out = new DataOutputStream(pythonSocket.getOutputStream());
        in = new DataInputStream(pythonSocket.getInputStream());
        LOG.debug("python socket was created successfully!");
    }

    public void writeOut(byte[] output) throws IOException {
        byte[] length = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(output.length).array();
        out.write(length);
        out.write(output);
        LOG.debug("Wrote out {} bytes to python", output.length);
    }

    public byte[] readIn() throws IOException {
        int length = in.readInt();
        LOG.debug("Attempting to Read in {} bytes from python", length);
        byte[] array = new byte[length];
        int success = in.read(array);
        LOG.debug("Read finished with result: {}", success);
        return array;
    }
}
