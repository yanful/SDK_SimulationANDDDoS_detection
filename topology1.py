from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI

class SDNTopo(Topo):
    def build(self):
        # Add switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        
        # Add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')
        
        # Connect hosts and switches
        self.addLink(h1, s1)
        self.addLink(h2, s2)
        self.addLink(h3, s2)
        self.addLink(h4, s3)
        self.addLink(s1, s2)
        self.addLink(s2, s3)

def start_network():
    topo = SDNTopo()
    net = Mininet(topo=topo, controller=RemoteController, link=TCLink)
    net.start()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    start_network()
