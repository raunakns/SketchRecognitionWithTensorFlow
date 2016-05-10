package connection;

import java.io.*;
import java.net.Socket;

/**
 * Created by david on 5/8/16.
 */
public class SocketConnection {

    Socket echoSocket;
    DataOutputStream out;
    DataInputStream in;

    public void createSocket(int portNumber) throws IOException {
        echoSocket = new Socket("localhost", portNumber);
        out = new DataOutputStream(echoSocket.getOutputStream());
        in = new DataInputStream(echoSocket.getInputStream());
    }

    public void writeOut(byte[] output) throws IOException {
        out.write(output.length);
        out.write(output);
    }

    public byte[] readIn() throws IOException {
        int length = in.readInt();
        byte[] array = new byte[length];
        int success = in.read(array);
        return array;
    }
}
