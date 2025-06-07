import React from 'react'
import { Link } from 'react-router-dom'
import {motion} from 'motion/react'
import Button from 'react-bootstrap/Button'
import Img1 from '../assets/Img1.jpg' 

const Hero = () => {
  return (
    <div className='custom-bg'>
      <div className="hero-section container-fluid px-4" style={{padding: '200px 0'}}>
        <div className="row align-items-center">
          <motion.div 
          initial={{opacity: 0}}
          animate={{opacity: 1}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
          className="col-md-6">
            <p className='display-5 fw-bold text-color'>Unmask Manipulated Media with AI Powered Precision</p>
            <p className='text-mute fs-5'>Our advanced AI technology helps you identify manipulated videos with industry-leading accuracy.</p>
            <Button variant="primary" className='text-dark px-5 fw-bold mt-2'><Link to='/predict' className='text-decoration-none fw-bold text-dark'>Try it &rarr;</Link></Button>
            <Button variant='outline-primary' className='text-white px-5 fw-bold mt-2 ms-3'><a href='#work' className='text-decoration-none fw-bold text-white'>How it works</a></Button>
          </motion.div>
          <motion.div
          initial={{opacity: 0}}
          animate={{opacity: 1}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
          className="col-md-6 text-center">
            <img src={Img1} alt="Failed to load image" className='img-fluid' style={{maxHeight: '400px'}} />
          </motion.div>
        </div>
      </div>

      <div id="work" className="card-contain-color" style={{padding: '100px 0'}}>
        <motion.h1
        initial={{opacity: 0}}
        whileInView={{opacity: 1}}
        viewport={{amount: 0.4, once: true}}
        transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
        className='text-color text-center'>How It Works</motion.h1>

        <motion.p
        initial={{opacity: 0}}
        whileInView={{opacity: 1}}
        viewport={{amount: 0.4, once: true}}
        transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
        className='text-center text-mute fs-5'>Our advanced AI technology analyzes videos frame by frame to detect manipulations</motion.p>
        <div className="py-5 d-flex align-items-center justify-content-center container">
          <motion.div
          initial={{y: -100, opacity: 0}}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{amount: 0.4, once: true}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
          className="col-md-4 text-center card1">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video h-8 w-8 text-blue-600 dark:text-blue-300 fs-4 text-color"><path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"></path><rect x="2" y="6" width="14" height="12" rx="2"></rect></svg>
            </span>
            <p className='text-white fw-bold fs-4'>Upload Video</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Upload any video file for analysis using our secure drag and drop interface</p>
          </motion.div>

          <motion.div
          initial={{y: -100, opacity: 0}}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{amount: 0.4, once: true}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
          className="col-md-4 text-center card2">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className="text-white fs-3 d-inline-block"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path></svg>
            </span>
            <p className='text-white fw-bold fs-4'>AI Analysis</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Our advanced AI model analyzes the video for signs of manipulation</p>
          </motion.div>

          <motion.div
          initial={{y: -100, opacity: 0}}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{amount: 0.4, once: true}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}} 
          className="col-md-4 text-center card2">
            <span className="px-3 py-3 bg-primary rounded-circle d-inline-block">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chart-column h-8 w-8 text-white"><path d="M3 3v16a2 2 0 0 0 2 2h16"></path><path d="M18 17V9"></path><path d="M13 17V5"></path><path d="M8 17v-3"></path></svg>
            </span>
            <p className='text-white fw-bold fs-4'>Get Results</p>
            <p className='text-mute text-center' style={{marginTop: '-15px'}}>Receive detailed analysis with confidence scores and visual indicators</p>
          </motion.div>

        </div>
      </div>

      <div id="features" style={{padding: '120px 0'}}>
        <motion.h1
        initial={{opacity: 0}}
        whileInView={{opacity: 1}}
        viewport={{amount: 0.4, once: true}}
        transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
        className='text-color text-center'>Key Features</motion.h1>

        <motion.p
        initial={{opacity: 0}}
        whileInView={{opacity: 1}}
        viewport={{amount: 0.4, once: true}}
        transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}}
        className='text-center text-mute fs-5 mb-5'>Our platform offers comprehensive deepfake detection capabilities</motion.p>

        <div className="container">
          <div className="row g-4">
            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Face Manipulation Detection</h3>
                <p className="text-mute">Identify facial manipulations with high precision</p>
                <p className="text-white">Our model can detect subtle inconsistencies in facial features, lighting, and expressions that indicate manipulation.</p>
              </div>
            </motion.div>

            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5, delay: 0.2}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Full-Body Detection</h3>
                <p className="text-mute">Analyze body movements and inconsistencies</p>
                <p className="text-white">Our system examines body movements, proportions, and physics to identify full-body deepfakes and manipulations.</p>
              </div>
            </motion.div>

            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5, delay: 0.4}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Metadata Analysis</h3>
                <p className="text-mute">Examine digital fingerprints in media files</p>
                <p className="text-white">We analyze file metadata to identify signs of editing, compression artifacts, and other technical indicators of manipulation.</p>
              </div>
            </motion.div>

            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Real-time Processing</h3>
                <p className="text-mute">Get results quickly with our optimized pipeline</p>
                <p className="text-white">Our advanced processing system ensures fast and accurate analysis while keeping users informed throughout the evaluation process.</p>
              </div>
            </motion.div>

            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Heatmap Visualization</h3>
                <p className="text-mute">Understand model focus across video frames</p>
                <p className="text-white">Our system generates visual attention heatmap to highlight where the model is focusing in each frame.Provides Insight into manipulation detection.</p>
              </div>
            </motion.div>

            <motion.div 
            initial={{opacity: 0, y: 50}}
            whileInView={{opacity: 1, y: 0}}
            viewport={{amount: 0.4, once: true}}
            transition={{duration: 0.5, delay: 0.4}}
            className="col-md-4">
              <div className="p-4 rounded-3" style={{backgroundColor: '#0F1729', border: '1px solid #1E40AF'}}>
                <h3 className="text-color">Comprehensive Reports</h3>
                <p className="text-mute">Detailed analysis with visual explanations</p>
                <p className="text-white">Receive detailed reports with confidence scores, highlighted areas of concern, and frame-by-frame analysis.</p>
              </div>
            </motion.div>
          </div>

        </div>
      </div>

      <div id="message" style={{padding: '150px 0', backgroundColor: '#0F1729'}}>
        <div className="container-fluid px-4 d-flex justify-content-center align-items-center">
          <motion.div
          initial={{opacity: 0}}
          whileInView={{opacity: 1}}
          viewport={{amount: 0.4, once: true}}
          transition={{duration: 1, delay: 0.4, ease: 'easeInOut'}} 
          className="col-12 col-md-6">
            <h1 className='text-color'>Get in Touch</h1>
            <p className='text-mute'>Have questions about our deepfake detection technology? <br />Contact us for more information.</p>
            <div>
              <span className='text-white'>
                <i class="fa-solid fa-envelope"></i>
                <p className='text-mute'>info@deepsight.co.pk</p>
              </span>

              <span className='text-white'>
                <i class="fa-solid fa-location-dot"></i>
                <p className='text-mute'>Lahore Pakistan</p>
              </span>

              <span className='text-white'>
                <i class="fa-solid fa-phone"></i>
                <p className='text-mute'>(042) 3311 1678</p>
              </span>
            </div>
          </motion.div>

          <motion.div
          initial={{opacity: 0, y: 50}}
          whileInView={{opacity: 1, y: 0}}
          viewport={{amount: 0.4, once: true}}
          transition={{duration: 0.5, delay: 0.4}} 
          className="col-12 col-md-6">
            <div className="card-3 p-4 rounded-3" style={{backgroundColor:'#0F1729', border: '1px solid blue'}}>
              <h3 className="mb-2" style={{color: '#93C5FD'}}>Send us a Message</h3>
              <p className='text-mute mb-4'>Fill out the form below and we'll get back to you soon.</p>
              <form>
                <div className="row g-3 mb-3">
                  <div className="col-md-6">
                    <label className="text-color mb-2">First name</label>
                    <input style={{backgroundColor: '#0F1729'}} type="text" className="form-control text-white border-secondary" placeholder="John" />
                  </div>

                  <div className="col-md-6">
                    <label className="text-color mb-2">Last name</label>
                    <input style={{backgroundColor: '#0F1729'}} type="text" className="form-control text-white border-secondary" placeholder="Doe" />
                  </div>
                </div>

                <div className="mb-3">
                  <label className="text-color mb-2">Email</label>
                  <input style={{backgroundColor: '#0F1729'}} type="email" className="form-control text-white border-secondary" placeholder="john.doe@example.com" />
                </div>

                <div className="mb-3">
                  <label className="text-color mb-2">Subject</label>
                  <input style={{backgroundColor: '#0F1729'}} type="text" className="form-control text-white border-secondary" placeholder="How can we help you?" />
                </div>

                <div className="mb-4">
                  <label className="text-color mb-2">Message</label>
                  <textarea style={{backgroundColor: '#0F1729'}} className="form-control text-white border-secondary" rows="4" placeholder="Your message here..."></textarea>
                </div>

                <button type="submit" className="btn btn-primary w-100">Send Message</button>
              </form>
            </div>
          </motion.div>

        </div>
      </div>
      
    </div>
  )
}

export default Hero